# graph_of_thoughts/context_manager.py

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from graph_of_thoughts.constants import (
    EMBEDDING_MODEL,
    IMPORTANCE_DECAY_FACTOR,
    MODEL_NAME,
    PRUNE_THRESHOLD,
    console,
)
from graph_of_thoughts.graph_components import (
    EmbeddingEngine,
    GraphStorage,
    ReasoningEngine,
    build_initial_graph,
)
from graph_of_thoughts.models import ChainOfThought, SeedData
from graph_of_thoughts.unified_llm import UnifiedLLM
from graph_of_thoughts.utils import (
    get_llm_model,
    get_sentence_transformer,
    get_tokenizer,
    get_unified_llm_model,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)  # Enable debug logs


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Let the base class handle anything else
        return super().default(obj)


class FastEmbeddingIndex:
    """Legacy FAISS index wrapper, kept for backward compatibility."""

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension)
        self.nodes = []

    def add_node(self, node_id, embedding):
        self.nodes.append(node_id)
        self.index.add(np.array([embedding], dtype=np.float32))

    def query(self, embedding, top_k=3) -> list:
        _, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            top_k,
        )
        return [self.nodes[i] for i in indices[0]]


# Assuming ChainOfThought is defined somewhere
# from your_module import ChainOfThought
import json
from typing import Any

# Assuming ChainOfThought is defined elsewhere
# from your_module import ChainOfThought


class ChainOfThoughtParser:
    """Helper class to extract and validate JSON from LLM output."""

    JSON_PATTERNS: list[str] = [
        r"<internal>\s*(\{.*?\})\s*</internal>",
        r"<json>\s*(\{.*?\})\s*</json>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
        # This pattern is kept as a fallback if it matches a chunk,
        # but we’ll use balanced extraction if it fails.
        r'(\{[\s\S]*?"nodes"[\s\S]*?"edges"[\s\S]*\})',
    ]

    def extract_json_string(self, raw_output: str) -> str:
        """Extract a JSON substring from the raw output using regex or balanced extraction.

        Args:
            raw_output: Raw string output from the LLM.

        Returns:
            Extracted JSON string.

        Raises:
            ValueError: If no valid JSON block is found.
        """
        # Try regex patterns first
        for pattern in self.JSON_PATTERNS:
            match = re.search(pattern, raw_output, re.DOTALL)
            if match:
                json_string = match.group(1).strip()
                console.log(f"✅ Found JSON using pattern: {pattern}", style="info")
                return json_string

        # Fallback: use balanced extraction from the first '{'
        try:
            balanced_json = self.extract_json_balanced(raw_output)
            console.log("✅ Extracted JSON using balanced extraction", style="info")
            return balanced_json
        except ValueError:
            pass

        console.log("❌ No valid JSON block found in LLM output!", style="warning")
        raise ValueError("No valid JSON block found.")

    def extract_json_balanced(self, text: str) -> str:
        """Extract a balanced JSON substring from text using a stack approach.

        This method finds the first '{' and then scans the text while counting
        opening and closing braces, respecting quoted strings.

        Args:
            text: Raw text to search.

        Returns:
            A substring representing a balanced JSON object.

        Raises:
            ValueError: If no balanced JSON object is found.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found.")
        count = 0
        in_string = False
        escape = False
        for i, char in enumerate(text[start:], start=start):
            if char == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                    if count == 0:
                        return text[start : i + 1]
            # Handle escape characters
            if char == "\\" and not escape:
                escape = True
            else:
                escape = False
        raise ValueError("No balanced JSON object found.")

    def load_json(self, json_string: str) -> dict[str, Any]:
        """Load JSON from a string, with a fallback to fix common formatting issues."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            console.log(
                f"❌ JSON Decode Error: {e}\nAttempting to fix JSON format...", style="warning"
            )
            fixed_json = re.sub(r"([{,])\s*(\w+):", r'\1"\2":', json_string)
            fixed_json = fixed_json.replace("'", '"')
            try:
                data = json.loads(fixed_json)
                console.log("🔧 Fixed JSON format issues and successfully parsed", style="info")
                return data
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format.") from e

    def validate_structure(self, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure the JSON has 'nodes' and 'edges'. If not, try to create a valid structure."""
        if "nodes" not in data or "edges" not in data:
            console.log(
                f"❌ Parsed JSON missing required fields: {list(data.keys())}", style="warning"
            )
            fixed_data = {"nodes": data.get("nodes", {}), "edges": data.get("edges", [])}
            if not fixed_data["nodes"] and not fixed_data["edges"]:
                raise ValueError("JSON missing 'nodes' or 'edges'.")
            console.log("🔧 Created valid structure from partial data", style="info")
            return fixed_data

        return data

    def validate_nodes(self, nodes: Any) -> dict[str, Any]:
        """Ensure nodes are in dictionary format and that all values are strings.

        Args:
            nodes: The nodes data extracted from JSON.

        Returns:
            A dictionary where every node value is a string.
        """
        if isinstance(nodes, dict):
            # Convert any non-string values to strings
            return {
                key: value if isinstance(value, str) else str(value) for key, value in nodes.items()
            }
        console.log(f"❌ 'nodes' is not a dictionary: {type(nodes)}", style="warning")
        if isinstance(nodes, list) and all(isinstance(n, dict) for n in nodes):
            nodes_dict = {}
            for i, node in enumerate(nodes):
                if "id" in node and "content" in node:
                    nodes_dict[node["id"]] = node["content"]
                elif "name" in node and "description" in node:
                    nodes_dict[node["name"]] = node["description"]
                else:
                    nodes_dict[f"node_{i}"] = str(node)
            console.log("🔧 Converted nodes list to dictionary", style="info")
            return nodes_dict
        return {}

    def validate_edges(self, edges: Any) -> list[list[str]]:
        """Ensure edges are returned as a list of two-element lists."""
        if not isinstance(edges, list):
            console.log(f"❌ 'edges' is not a list: {type(edges)}", style="warning")
            if isinstance(edges, dict):
                edges_list = []
                for source, targets in edges.items():
                    if isinstance(targets, list):
                        for target in targets:
                            edges_list.append([source, target])
                    else:
                        edges_list.append([source, str(targets)])
                console.log("🔧 Converted edges dict to list", style="info")
                edges = edges_list
            else:
                return []

        valid_edges = []
        for edge in edges:
            if isinstance(edge, list) and len(edge) == 2:
                valid_edges.append([str(edge[0]), str(edge[1])])
            elif isinstance(edge, dict) and "source" in edge and "target" in edge:
                valid_edges.append([str(edge["source"]), str(edge["target"])])
            else:
                console.log(f"❌ Skipping invalid edge format: {edge}", style="warning")
        return valid_edges

    def parse(self, raw_output: str) -> "ChainOfThought":
        """Orchestrate extraction, parsing, and validation to return a ChainOfThought."""
        json_string = self.extract_json_string(raw_output)
        console.log(
            f"✅ Extracted JSON String (first 100 chars):\n{json_string[:100]}...", style="context"
        )
        data = self.load_json(json_string)
        data = self.validate_structure(data)
        data["nodes"] = self.validate_nodes(data["nodes"])
        data["edges"] = self.validate_edges(data["edges"])
        console.log(
            f"✅ Successfully parsed chain of thought with {len(data['nodes'])} nodes and {len(data['edges'])} edges",
            style="info",
        )
        return ChainOfThought(nodes=data["nodes"], edges=data["edges"])


def parse_chain_of_thought(raw_output: str) -> "ChainOfThought":
    """
    Extracts and validates structured JSON from LLM output, ensuring robustness
    for both HF and llama_cpp backends.

    Args:
        raw_output: The raw string output from the LLM

    Returns:
        ChainOfThought object with nodes and edges

    Raises:
        ValueError: If no valid JSON is found or JSON is missing required fields
    """
    return ChainOfThoughtParser().parse(raw_output)


class ContextGraphManager:
    """
    Manages a dynamic context graph using specialized components.
    """

    def __init__(
        self,
        tokenizer: Optional["AutoTokenizer"] = None,
        model: Optional["AutoModelForCausalLM"] = None,
        initial_graph: nx.DiGraph | None = None,
        sentence_model: SentenceTransformer | None = None,
    ):
        # Initialize components
        # Use the imported build_initial_graph function
        if initial_graph is None:
            initial_graph = build_initial_graph()
        console.log(f"Initializing graph storage with {len(initial_graph)} nodes.")
        self.sentence_model = sentence_model or get_sentence_transformer(
            model_name=EMBEDDING_MODEL,
        )
        self.graph_storage = GraphStorage(
            initial_graph=initial_graph,
        )

        self.embedding_engine = EmbeddingEngine(sentence_model=sentence_model)
        self.reasoning_engine = ReasoningEngine(
            graph_storage=self.graph_storage,
            embedding_engine=self.embedding_engine,
        )

        # For backward compatibility
        self.graph = self.graph_storage.graph

        # Keep model components for generation
        self.tokenizer = tokenizer or get_tokenizer()
        self.model = model or get_llm_model()
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.tokenizer = tokenizer  # For HF models; may be None for llama_cpp backend

        # Initialize sentence model if provided or use default

    def add_context(
        self,
        node_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Add content to the graph and index it for retrieval."""
        self.graph_storage.add_node(node_id, content, metadata)
        self.embedding_engine.add_node_embedding(node_id, content)

    def query_context(self, query: str, top_k: int = 3) -> list[str]:
        """Find the most relevant nodes for a query."""
        return self.embedding_engine.find_similar_nodes(query, top_k)

    def visualize_graph_as_text(self, max_nodes=8) -> str:
        """Generate a focused text representation of the graph."""
        text = ["📌 **Current Knowledge Graph**\n"]

        # Get nodes with importance values
        nodes_with_importance = []
        for node_id in self.graph_storage.graph.nodes():
            data = self.graph_storage.graph.nodes[node_id].get("data", {})
            importance = 0.0

            # Handle different data structures
            if isinstance(data, dict):
                importance = data.get("importance", 0.0)
            elif hasattr(data, "metadata") and isinstance(data.metadata, dict):
                importance = data.metadata.get("importance", 0.0)

            nodes_with_importance.append((node_id, importance))

        # Sort by importance (descending) and take top N
        important_nodes = [
            n[0]
            for n in sorted(nodes_with_importance, key=lambda x: x[1], reverse=True)[:max_nodes]
        ]

        # Add nodes
        text.append("🟢 **Nodes**:")
        for node in important_nodes:
            content = self.graph_storage.get_node_content(node) or "No description"
            if len(content) > 50:  # Truncate long content
                content = content[:47] + "..."
            text.append(f"  - {node}: {content}")

        # Add edges between important nodes
        text.append("\n🔗 **Edges**:")
        edge_count = 0
        for source, target in self.graph_storage.graph.edges():
            if source in important_nodes and target in important_nodes:
                text.append(f"  - {source} → {target}")
                edge_count += 1

        if edge_count == 0:
            text.append("  (No edges between shown nodes)")

        return "\n".join(text)

    def graph_to_json(self) -> dict:
        """Convert the graph storage to a JSON-serializable format."""
        try:
            # Get the raw graph data
            graph_data = self.graph_storage.to_json()

            # Process nodes to ensure all data is serializable
            for node in graph_data.get("nodes", []):
                if "data" in node and isinstance(node["data"], dict):
                    data = node["data"]

                    # Handle datetime objects
                    if "created_at" in data and isinstance(data["created_at"], datetime):
                        data["created_at"] = data["created_at"].isoformat()

                    # Handle other potential non-serializable objects
                    for key, value in data.items():
                        if hasattr(value, "model_dump"):
                            data[key] = value.model_dump()

            return graph_data
        except Exception as e:
            console.print(f"[Error] Failed to convert graph to JSON: {e}", style="warning")
            # Return a minimal valid structure
            return {"nodes": [], "edges": []}

    def save_graph_state(self, filepath):
        """Save the current graph state to a file."""
        try:
            graph_data = self.graph_to_json()

            with open(filepath, "w") as f:
                json.dump(graph_data, f, cls=DateTimeEncoder, indent=2)

            console.print(f"Graph state saved to {filepath}", style="info")
            return True
        except Exception as e:
            console.print(f"[Error] Failed to save graph state: {e}", style="error")
            return False

    def decay_importance(
        self,
        decay_factor: float = IMPORTANCE_DECAY_FACTOR,
        adaptive: bool = True,
    ) -> None:
        """Apply decay to all nodes."""
        for node_id in self.graph_storage.graph.nodes():
            self.graph_storage.decay_node_importance(
                node_id=node_id,
                decay_factor=decay_factor,
            )
        return

    def prune_context(self, threshold: float = PRUNE_THRESHOLD, max_nodes: int = 20) -> None:
        """Remove low-importance nodes and limit total graph size."""
        # First, prune by importance threshold
        self.graph_storage.prune_low_importance_nodes(threshold)

        # Then, if still too large, keep only the most important nodes
        if len(self.graph_storage.graph.nodes) > max_nodes:
            nodes_with_importance = []
            for node_id in self.graph_storage.graph.nodes():
                node_data = self.graph_storage.graph.nodes[node_id].get("data", {})
                importance = node_data.get("importance", 0.0)
                nodes_with_importance.append((node_id, importance))

            # Sort by importance (descending)
            nodes_with_importance.sort(key=lambda x: x[1], reverse=True)

            # Remove least important nodes beyond our limit
            nodes_to_remove = [node_id for node_id, _ in nodes_with_importance[max_nodes:]]
            for node_id in nodes_to_remove:
                self.graph_storage.remove_node(node_id)

    def iterative_refinement(self, reasoning_output: str | dict) -> None:
        """Update the graph based on structured reasoning output."""
        try:
            # Handle different input types
            if isinstance(reasoning_output, dict):
                # Convert dictionary to ChainOfThought
                chain_obj = ChainOfThought(
                    nodes=reasoning_output.get("nodes", {}),
                    edges=reasoning_output.get("edges", []),
                )
            elif isinstance(reasoning_output, str):
                if reasoning_output.strip() == "{}":
                    console.log("[Warning] Empty JSON structure received", style="warning")
                    return
                chain_obj = parse_chain_of_thought(reasoning_output)
            else:
                console.log(
                    f"[Error] Unexpected reasoning_output type: {type(reasoning_output)}",
                    style="warning",
                )
                return

            # Ensure we have the required fields
            if not chain_obj.nodes and not chain_obj.edges:
                console.log("[Warning] Missing required fields in JSON", style="warning")
                if isinstance(reasoning_output, dict):
                    console.print_json(data=reasoning_output)
                else:
                    console.log(reasoning_output)
                return

            # Only update if we have meaningful content
            if chain_obj.nodes or chain_obj.edges:
                self.reasoning_engine.update_from_chain_of_thought(chain_obj)
                console.log(
                    f"[Success] Updated graph with {len(chain_obj.nodes)} nodes and {len(chain_obj.edges)} edges",
                    style="info",
                )
            else:
                console.log("[Warning] No nodes or edges to add from reasoning", style="warning")

        except Exception as e:
            console.log(f"[Error] Failed to process reasoning output: {e}", style="warning")


def get_context_mgr(
    model_name: str = MODEL_NAME,
    unified_model: UnifiedLLM | None = None,
) -> ContextGraphManager:
    """
    Create and initialize a ContextGraphManager.
    If 'unified_model' is provided, it is used directly; otherwise, the default HF model/tokenizer are loaded.
    """
    if unified_model is not None:
        model = unified_model
        # For UnifiedLLM instances, we use the built-in tokenizer functionality
        # which is either the actual tokenizer (HF) or the UnifiedLLM itself (llama_cpp)
        tokenizer = unified_model.tokenizer
    else:
        # Load a new unified model with HF backend by default
        model = get_unified_llm_model(backend="hf", model_name=model_name)
        tokenizer = model.tokenizer

    return ContextGraphManager(tokenizer=tokenizer, model=model)


def seed_nodes(context_manager: ContextGraphManager, seed_data: list[SeedData]) -> None:
    """Add seed data to the context graph."""
    for data in seed_data:
        context_manager.add_context(data.node_id, data.content, data.metadata)


def simulate_chat(
    context_manager: ContextGraphManager,
    conversation_inputs: list[str],
    seed_data: list[SeedData] | None = None,
    experiment_name: str = "default_experiment",
) -> None:
    """Simulation function that uses ChatManager for consistency."""
    from graph_of_thoughts.chat_manager import ChatManager

    # Initialize the chat manager with our context manager
    chat_manager = ChatManager(context_manager)

    # Add seed data if provided
    if seed_data:
        seed_nodes(context_manager, seed_data)

    # Run the simulation using the chat manager
    experiment_data = chat_manager.simulate_conversation(
        inputs=conversation_inputs, experiment_name=experiment_name
    )

    console.log(f"Experiment completed with {len(experiment_data)} turns", style="info")


if __name__ == "__main__":
    from graph_of_thoughts.chat_manager import ChatManager

    # Initialize context manager
    context_manager = get_context_mgr()
    chat_manager = ChatManager(context_manager=context_manager)

    # Define conversation
    canned_conversation = [
        "How can I improve the caching mechanism without increasing latency?",
        "What strategies can reduce database load during peak hours?",
        "Can we optimize the cache eviction policy for better performance?",
        "What are some best practices for scaling the caching system?",
    ]

    # Run conversation simulation
    experiment_data = chat_manager.simulate_conversation(
        inputs=canned_conversation,
    )

    # Print graph summary
    console.log(
        "\n[Final Context Graph]:",
        context_manager.graph_storage.visualize_as_text(),
        style="context",
    )
    console.log("[Info] Experiment completed.", style="info")
