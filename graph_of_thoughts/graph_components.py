# graph_of_thoughts/graph_components.py

from datetime import datetime, timezone
from typing import Self

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from graph_of_thoughts.constants import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    IMPORTANCE_DECAY_FACTOR,
    PRUNE_THRESHOLD,
    console,
)
from graph_of_thoughts.models import ChainOfThought, ContextNode, Node
from graph_of_thoughts.utils import get_sentence_transformer


def build_initial_graph() -> nx.DiGraph:
    """
    Creates a small directed (hierarchical) graph with some default edges and attributes.
    """
    graph = nx.DiGraph()

    # root node
    node_root = ContextNode(
        node_id="root",
        content="Top-level concept",
        metadata={"importance": 1.0},
    )
    graph.add_node("root", data=node_root)

    # subA node
    node_sub_a = ContextNode(
        node_id="subA",
        content="A sub concept under root",
        metadata={"importance": 0.9},
    )
    graph.add_node("subA", data=node_sub_a)

    # subB node
    node_sub_b = ContextNode(
        node_id="subB",
        content="Another sub concept under root",
        metadata={"importance": 0.9},
    )
    graph.add_node("subB", data=node_sub_b)

    # edges
    graph.add_edge("root", "subA")
    graph.add_edge("root", "subB")

    return graph


class GraphStorage:
    """
    Handles storage, serialization, and basic operations for the knowledge graph.
    """

    def __init__(self, initial_graph: nx.DiGraph | None = None):
        """Initialize with an optional existing graph."""
        self.graph = initial_graph if initial_graph is not None else nx.DiGraph()

    def add_node(
        self,
        node_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a node to the graph with content and metadata."""
        if metadata is None:
            metadata = {"importance": 1.0}

        node = Node(
            content=content,
            importance=metadata.get("importance", 1.0),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.graph.add_node(node_id, data=node.model_dump())
        console.log(f"Added node: {node_id}", style="info")

    def add_edge(self, source: str, target: str) -> None:
        """Add a directed edge between nodes."""
        if source in self.graph.nodes and target in self.graph.nodes:
            self.graph.add_edge(source, target)
            console.log(f"Added edge: {source} → {target}", style="info")
        else:
            missing = [n for n in (source, target) if n not in self.graph.nodes]
            console.log(f"Cannot add edge - missing nodes: {missing}", style="warning")

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges from the graph."""
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            console.log(f"Removed node: {node_id}", style="info")

    def get_node_content(self, node_id: str) -> str | None:
        """Get the content of a node."""
        if node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id].get("data", {})
            # Handle both dictionary and Node object formats
            if isinstance(node_data, dict):
                return node_data.get("content")
            elif hasattr(node_data, "content"):
                return node_data.content
        return None

    def to_json(self) -> dict:
        """Convert the graph to a JSON-serializable format."""
        data = nx.node_link_data(self.graph)
        # Convert node data to serializable format
        for node in data["nodes"]:
            for key, value in node.items():
                if hasattr(value, "model_dump"):
                    node[key] = value.model_dump()
        return data

    def visualize_as_text(self) -> str:
        """Generate a text representation of the graph."""
        text = ["📌 **Current Knowledge Graph**\n"]

        # Add nodes
        text.append("🟢 **Nodes**:")
        for node, data in self.graph.nodes(data=True):
            content = data.get("data", {}).get("content", "No description")
            if len(content) > 50:  # Truncate long content
                content = content[:47] + "..."
            text.append(f"  - {node}: {content}")

        # Add edges
        text.append("\n🔗 **Edges**:")
        for source, target in self.graph.edges():
            text.append(f"  - {source} → {target}")

        return "\n".join(text)

    def decay_node_importance(
        self,
        node_id: str,
        decay_factor: float = IMPORTANCE_DECAY_FACTOR,
    ) -> Self:
        """Apply time-based decay to a node's importance."""
        if node_id not in self.graph.nodes:
            return self

        node_data = self.graph.nodes[node_id].get("data", {})
        if not node_data:
            return self

        # Get current importance
        importance = node_data.get("importance", 1.0)

        # Get node age - handle string or datetime objects
        created_at_str = node_data.get("created_at")
        if not created_at_str:
            # No timestamp available, can't decay
            return self

        try:
            # Handle both string and datetime objects
            if isinstance(created_at_str, str):
                created_at = datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                )
            elif isinstance(created_at_str, datetime):
                created_at = created_at_str
            else:
                # Unsupported type
                console.log(
                    f"Unsupported created_at type for node {node_id}: {type(created_at_str)}",
                    style="warning",
                )
                return self

            # Calculate decay
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            decay_rate = decay_factor ** (age_seconds / 3600)  # Per hour

            # Update importance
            new_importance = importance * decay_rate
            node_data["importance"] = new_importance
            self.graph.nodes[node_id]["data"] = node_data
        except Exception as e:
            console.log(f"Error during decay for node {node_id}: {e}", style="warning")

        return self

    def prune_low_importance_nodes(self, threshold: float = PRUNE_THRESHOLD) -> None:
        """Remove nodes with importance below the threshold."""
        to_remove = []

        for node_id in self.graph.nodes():
            # Apply decay first
            self.decay_node_importance(node_id)

            # Check if below threshold
            node_data = self.graph.nodes[node_id].get("data", {})
            importance = node_data.get("importance", 1.0)

            if importance < threshold:
                to_remove.append(node_id)

        # Remove identified nodes
        for node_id in to_remove:
            self.remove_node(node_id)

        if to_remove:
            console.log(f"Pruned {len(to_remove)} low-importance nodes", style="info")
        return


class EmbeddingEngine:
    """
    Handles node embeddings, semantic similarity, and context retrieval.
    """

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        sentence_model: SentenceTransformer | None = None,
    ):
        """Initialize the embedding engine with a FAISS index."""
        self.sentence_model = sentence_model or get_sentence_transformer(
            model_name=EMBEDDING_MODEL,
        )
        self.index = faiss.IndexFlatL2(dimension)
        self.nodes = []  # Parallel array to track node IDs

    def add_node_embedding(self, node_id: str, content: str) -> None:
        """Compute and add a node's embedding to the index."""
        embedding = self.sentence_model.encode(content, convert_to_numpy=True)
        self.nodes.append(node_id)
        self.index.add(np.array([embedding], dtype=np.float32))

    def find_similar_nodes(self, query: str, top_k: int = 3) -> list[str]:
        """Find nodes most similar to the query."""
        if not self.nodes:  # Empty index
            return []

        # Encode query
        query_emb = self.sentence_model.encode(query, convert_to_numpy=True)

        # Search index
        distances, indices = self.index.search(
            np.array([query_emb], dtype=np.float32),
            min(top_k, len(self.nodes)),  # Don't request more than we have
        )

        # Convert indices to node IDs
        result = [self.nodes[i] for i in indices[0]]
        console.log(f"Query: '{query}'", style="context")
        console.log(f"Similar nodes: {result}", style="context")

        return result

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings = self.sentence_model.encode([text1, text2], convert_to_numpy=True)
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity


class ReasoningEngine:
    """
    Processes reasoning chains and manages iterative refinement of the graph.
    """

    def __init__(
        self,
        graph_storage: GraphStorage,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        """Initialize with references to graph and embedding components."""
        self.graph = graph_storage
        self.embeddings = embedding_engine

    def update_from_chain_of_thought(
        self,
        chain: ChainOfThought,
    ) -> None:
        """Update the graph based on a reasoning chain."""
        # Add nodes
        for node_id, description in chain.nodes.items():
            # Create a unique ID to avoid conflicts
            unique_id = f"reason_{node_id}"

            # Add to graph storage
            self.graph.add_node(unique_id, description)
            self.embeddings.add_node_embedding(unique_id, description)

        # Add edges
        for source, target in chain.edges:
            # Convert to our prefixed node IDs
            source_id, target_id = f"reason_{source}", f"reason_{target}"
            self.graph.add_edge(source_id, target_id)
