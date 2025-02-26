# graph_of_thoughts/chat_manager.py

import json
import os
import re
from datetime import datetime
from typing import Any

import networkx as nx
import torch
from transformers import GenerationConfig

from graph_of_thoughts.constants import MAX_NEW_TOKENS, OUTPUT_DIR, console
from graph_of_thoughts.context_manager import ContextGraphManager, seed_nodes
from graph_of_thoughts.evaluate_llm_graph import GraphMetrics, KnowledgeGraph
from graph_of_thoughts.models import SeedData


def extract_sections(response: str) -> tuple[str, str]:
    """
    Extract the internal JSON block and the final answer from the response.
    Expected format:
        ... <internal> ... </internal> ... <final> ... </final> ...
    """
    internal_match = re.search(r"<internal>(.*?)</internal>", response, re.DOTALL)
    final_match = re.search(r"<final>(.*?)</final>", response, re.DOTALL)

    # Ensure extracted JSON is valid
    internal = internal_match.group(1).strip() if internal_match else "{}"
    final = final_match.group(1).strip() if final_match else response.strip()

    try:
        json.loads(internal)  # Check if valid JSON
    except ValueError as e:
        console.log(
            f"[Error] Extracted JSON is invalid. Using fallback: {e}",
            style="warning",
        )
        internal = "{}"

    return internal, final


class ChatManager:
    """
    Manages conversations with the LLM and updates the context graph.
    """

    def __init__(self, context_manager: ContextGraphManager):
        """Initialize with a context manager."""
        self.context_manager = context_manager

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        # Get relevant nodes
        relevant_nodes = self.context_manager.query_context(query, top_k=3)
        navigation_path = " → ".join(relevant_nodes)
        # Construct prompt with two distinct sections
        extended_prompt = f"""
[Knowledge Graph Navigation]
- Your goal is to expand relevant concepts based on the **existing graph**.
- Navigate down from the **root concept** to related subtopics.
- Prioritize depth over breadth—go deeper before adding new high-level topics.

[Current Query]: {query}
[Current Navigation Path]: {navigation_path}
[Existing Graph Structure]:
{self.context_manager.visualize_graph_as_text()}

[Reasoning Instructions]:
1️⃣ Identify missing knowledge gaps in the structure.
2️⃣ Expand deeper where necessary—**do not just add random new nodes.**
3️⃣ Maintain a logical structure using causality and dependencies.
4️⃣ Output only **valid JSON inside <internal>...</internal>** tags for graph updates.
5️⃣ Then, output the final answer for the user inside <final>...</final> tags.

[Generated Knowledge Graph Update]:
"""
        # TODO: Remove
        # console.log(f"Prompt: {extended_prompt}", style="prompt")
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
        return self.context_manager.tokenizer.decode(output[0], skip_special_tokens=True)

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
        response = self.generate_response(user_input)

        # Extract internal reasoning (for graph update) and final answer (for the user)
        internal_json, final_response = extract_sections(response)
        console.log(f"[Debug] Extracted JSON:\n{internal_json}", style="warning")

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
