# graph_of_thoughts/chat_manager.py

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import networkx as nx
import torch
from transformers import GenerationConfig

from graph_of_thoughts.constants import MAX_NEW_TOKENS, OUTPUT_DIR, console
from graph_of_thoughts.context_manager import ContextGraphManager, seed_nodes
from graph_of_thoughts.evaluate_llm_graph import GraphMetrics, KnowledgeGraph
from graph_of_thoughts.models import SeedData
from graph_of_thoughts.utils import build_llama_instruct_prompt


def extract_sections(response: str) -> tuple[str, str]:
    """
    Extract JSON block and final answer, handling template echoing.
    """
    console.log(f"[Debug] Raw LLM Response (first 300 chars):\n{response[:300]}...", style="dim")

    # Check for template echoing
    if "ConceptName1" in response and "Description of concept 1" in response:
        console.log("[Warning] LLM echoed the template instead of filling it", style="warning")
        internal = "{}"
    else:
        # Try to extract using <internal> tags
        internal_match = re.search(r"<internal>\s*(\{.*?\})\s*</internal>", response, re.DOTALL)

        if internal_match:
            internal = internal_match.group(1).strip()
        else:
            # Try other formats
            json_match = re.search(
                r"<json>\s*(\{.*?\})\s*</json>", response, re.DOTALL
            ) or re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)

            if json_match:
                internal = json_match.group(1).strip()
            else:
                console.log("[Warning] No JSON block found in response", style="warning")
                internal = "{}"

    # Extract final response
    final_match = re.search(r"<final>\s*(.*?)\s*</final>", response, re.DOTALL)
    if final_match:
        final = final_match.group(1).strip()
    else:
        # Use everything except JSON blocks
        final = re.sub(r"<internal>.*?</internal>", "", response, flags=re.DOTALL)
        final = re.sub(r"<json>.*?</json>", "", final, flags=re.DOTALL)
        final = re.sub(r"```(?:json)?.*?```", "", final, flags=re.DOTALL)
        final = final.strip()

    # Ensure we have valid JSON
    try:
        if internal != "{}":
            json_data = json.loads(internal)
            if "nodes" not in json_data or "edges" not in json_data:
                json_data = {"nodes": {}, "edges": []}
                internal = json.dumps(json_data)
    except json.JSONDecodeError:
        internal = "{}"

    return internal, final


class ChatManager:
    """
    Manages conversations with the LLM and updates the context graph.
    """

    def __init__(self, context_manager: ContextGraphManager):
        """Initialize with a context manager."""
        self.context_manager = context_manager
        self.response_cache = {}  # query -> response cache

    @staticmethod
    def _get_nav_instructions(
        query: str,
        navigation_path,
        graph_structure,
    ) -> str:
        """
        Constructs a prompt for the user to continue the conversation.
        """
        # Construct prompt with two distinct sections
        # former:
        """[Knowledge Graph Navigation] - Your goal is to expand relevant concepts based on the **existing graph**. - Navigate down from the **root concept** to related subtopics. - Prioritize depth over breadth—go deeper before adding new high-level topics."""
        return f"""

[Current Query]: {query}
[Current Navigation Path]: {navigation_path}
[Existing Graph Structure]:
{graph_structure}

[RESPONSE FORMAT - FOLLOW EXACTLY]:
Your response MUST contain BOTH of these sections in this exact order:

1. FIRST, a JSON structure inside <internal> tags following this format:
<internal>
{{
  "nodes": {{
    "ConceptName1": "Description of concept 1",
    "ConceptName2": "Description of concept 2"
  }},
  "edges": [
    ["ConceptName1", "ConceptName2"]
  ]
}}
</internal>

2. SECOND, your answer to the user's question inside <final> tags:
<final>
Your detailed answer to the user's question goes here.
</final>
"""

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """
        Constructs an extended reasoning prompt where the LLM explicitly navigates the graph.
        Uses caching for improved performance on repeated queries.
        """
        # Check cache first
        if not hasattr(self, "response_cache"):
            self.response_cache = {}

        cache_key = hash(query)
        if cache_key in self.response_cache:
            console.log(f"[Cache] Using cached response for query: '{query[:30]}...'", style="info")
            return self.response_cache[cache_key]

        # Get relevant nodes
        relevant_nodes = self.context_manager.query_context(query, top_k=3)
        navigation_path = " → ".join(relevant_nodes)
        navigation_instructions = self._get_nav_instructions(
            query=query,
            navigation_path=navigation_path,
            graph_structure=self.context_manager.visualize_graph_as_text(),
        )
        user_text = f"{navigation_instructions}\n\nQuestion: {query}"
        system_prompt = """You are a knowledge graph builder. Create nodes and connections for the query.
First provide JSON in <internal> tags, then answer the question in <final> tags."""

        extended_prompt = build_llama_instruct_prompt(
            system_text=system_prompt,
            user_text=user_text,
        )

        # Tokenize and generate
        inputs = self.context_manager.tokenizer(extended_prompt, return_tensors="pt").to(
            self.context_manager.model.device
        )

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.context_manager.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output = self.context_manager.model.generate(
                **inputs, generation_config=generation_config
            )

        response = self.context_manager.tokenizer.decode(output[0], skip_special_tokens=True)

        # Store in cache for future use
        self.response_cache[cache_key] = response

        return response

    def process_turn(self, user_input: str, conversation_turn: int) -> str:
        """Process a single conversation turn."""
        # Decay node importance
        self.context_manager.decay_importance(decay_factor=0.95, adaptive=True)

        # Log user input and add to context
        console.log(f"[User {conversation_turn}]: {user_input}", style="user")
        self.context_manager.add_context(f"user_{conversation_turn}", user_input)

        # Retrieve relevant context for logging
        retrieved_context = self.context_manager.query_context(user_input, top_k=3)
        console.log(f"[Context Retrieved]: {retrieved_context}", style="context")

        # Generate full response
        console.log(f"[Debug] Generating response for: {user_input[:50]}...", style="info")
        start_time = time.time()

        response = self.generate_response(user_input)
        generation_time = time.time() - start_time
        console.log(f"[Debug] Response generation took {generation_time:.2f}s", style="info")

        # Extract internal reasoning (for graph update) and final answer (for the user)
        internal_json, final_response = extract_sections(response)
        console.log(f"[Debug] Extracted JSON size: {len(internal_json)} characters", style="info")

        console.log(f"[LLM Final Response {conversation_turn}]: {final_response}", style="llm")

        # Add final answer to context
        self.context_manager.add_context(f"llm_{conversation_turn}", final_response)

        # Update graph with internal JSON reasoning
        try:
            reasoning_output = json.loads(internal_json)
            self.context_manager.iterative_refinement(reasoning_output)
        except json.JSONDecodeError as e:
            console.log(
                f"[Error] JSON parsing failed: {e}. Extracted JSON snippet:\n"
                f"```json\n{internal_json[:300]}...\n```",  # Truncate to avoid excessive output
                style="warning",
            )
        except Exception as e:
            console.log(
                f"[Error] Unexpected issue during structured reasoning extraction: {e}. Last 100 chars of response:\n"
                f"```{response[-100:]}```",
                style="error",
            )

        # Return the final response to match test expectations
        return final_response

    def simulate_conversation(
        self,
        inputs: list[str],
        seed_data: list[SeedData] | None = None,
        experiment_name: str = "default_experiment",
    ) -> list[dict[str, Any]]:
        """Simulate a full conversation with multiple turns."""
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Add seed data if provided
        if seed_data:
            seed_nodes(self.context_manager, seed_data)

        experiment_data = []
        conversation_turn = 1

        for user_input in inputs:
            console.log(f"\n[User {conversation_turn}]: {user_input}", style="user")

            # Capture graph state before processing
            graph_before = self.context_manager.graph_storage.graph.copy()

            # Process the turn
            response = self.process_turn(user_input, conversation_turn)

            # Capture graph state after processing
            graph_after = self.context_manager.graph_storage.graph.copy()

            # Get retrieved context
            retrieved_context = self.context_manager.query_context(user_input, top_k=3)

            # Save graphs for evaluation
            baseline_path = OUTPUT_DIR / f"{experiment_name}_baseline_graph.json"
            llm_path = OUTPUT_DIR / f"{experiment_name}_llm_graph.json"

            # Create baseline and LLM graphs
            baseline = KnowledgeGraph("Baseline")
            baseline.graph = graph_before
            baseline.save_graph_json(baseline_path)

            llm_graph = KnowledgeGraph("LLM")
            llm_graph.graph = graph_after
            llm_graph.save_graph_json(llm_path)

            # Convert graphs to format expected by GraphMetrics
            baseline_data = {
                "nodes": self._extract_nodes(graph_before),
                "edges": list(graph_before.edges()),
            }

            llm_data = {
                "nodes": self._extract_nodes(graph_after),
                "edges": list(graph_after.edges()),
            }

            # Calculate metrics using the new GraphMetrics class
            metrics_calculator = GraphMetrics(baseline_data, llm_data)
            metrics = metrics_calculator.compute_metrics()

            # Save turn data
            turn_data = {
                "turn": conversation_turn,
                "user_input": user_input,
                "llm_response": response,
                "retrieved_context": retrieved_context,
                "graph_before": self._serialize_graph(graph_before),
                "graph_after": self._serialize_graph(graph_after),
                "metrics": metrics,
            }

            experiment_data.append(turn_data)
            conversation_turn += 1

        # Save complete experiment data
        experiment_file = OUTPUT_DIR / f"{experiment_name}_data.json"
        with open(experiment_file, "w") as f:
            json.dump(experiment_data, f, indent=4, default=self._datetime_handler)

        console.log(f"Experiment data saved to {experiment_file}", style="info")
        return experiment_data

    def _extract_nodes(self, graph: nx.DiGraph) -> dict:
        """Extract node content from the graph for metrics calculation."""
        nodes = {}
        for node in graph.nodes:
            if "data" in graph.nodes[node] and "content" in graph.nodes[node]["data"]:
                nodes[node] = graph.nodes[node]["data"]["content"]
            else:
                nodes[node] = "No description"
        return nodes

    @staticmethod
    def _serialize_graph(graph: nx.DiGraph) -> dict:
        """Convert a graph to a serializable format."""
        data = nx.node_link_data(graph, edges="links")
        for node in data["nodes"]:
            for key, value in node.items():
                if hasattr(value, "model_dump"):
                    node[key] = value.model_dump()
        return data

    @staticmethod
    def _datetime_handler(obj):
        """Handle datetime serialization for JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
