# graph_of_thoughts/context_manager.py

import json
import torch
import networkx as nx
import numpy as np
import re
from typing import List, Optional, TYPE_CHECKING
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer
from graph_of_thoughts.models import SeedData, ChainOfThought

import faiss

from graph_of_thoughts.constants import (
    console,
    EMBEDDING_MODEL,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    IMPORTANCE_DECAY_FACTOR,
    PRUNE_THRESHOLD,
)
from graph_of_thoughts.graph_components import (
    GraphStorage,
    EmbeddingEngine,
    ReasoningEngine,
    build_initial_graph,
)

import logging
from graph_of_thoughts.utils import (
    extract_and_clean_json,
    get_llm_model,
    get_sentence_transformer,
    get_tokenizer,
)

if TYPE_CHECKING:
    from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)  # Enable debug logs


class FastEmbeddingIndex:
    """Legacy FAISS index wrapper, kept for backward compatibility."""

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension)
        self.nodes = []

    def add_node(self, node_id, embedding):
        self.nodes.append(node_id)
        self.index.add(np.array([embedding], dtype=np.float32))

    def query(self, embedding, top_k=3) -> list:
        distances, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            top_k,
        )
        return [self.nodes[i] for i in indices[0]]


def parse_chain_of_thought(raw_output: str) -> ChainOfThought:
    """
    Extracts and validates structured JSON from LLM output, ensuring robustness.
    """
    # 🔍 Extract JSON inside <json>...</json>, allowing for any whitespace or newlines
    console.log("🔍 Extracting JSON from raw output", style="info")
    json_regex_pat = re.compile(r"<json>\s*(\{.*?\})\s*</json>", re.DOTALL)
    if match := json_regex_pat.search(raw_output) is None:
        console.log("❌ No valid JSON block found in LLM output!", style="warning")
        raise ValueError("No valid JSON block found.")

    json_string = match.group(1).strip()
    # ✅ Debugging: Print extracted JSON before parsing
    console.log(f"✅ Extracted JSON String:\n{json_string}", style="context")

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        console.log(
            f"❌ JSON Decode Error: {e}\nRaw JSON Extracted:\n{json_string}",
            style="warning",
        )
        raise ValueError("Invalid JSON format.")

    if "nodes" not in data or "edges" not in data:
        console.log(f"❌ Parsed JSON missing required fields: {data}", style="warning")
        raise ValueError("JSON missing 'nodes' or 'edges'.")

    return ChainOfThought(nodes=data["nodes"], edges=data["edges"])


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
        self.model.eval()

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

    def visualize_graph_as_text(self) -> str:
        """Get a text representation of the graph."""
        return self.graph_storage.visualize_as_text()

    def graph_to_json(self) -> str:
        """Convert the graph to JSON format."""
        return self.graph_storage.to_json()

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

    def prune_context(self, threshold: float = PRUNE_THRESHOLD) -> None:
        """Remove low-importance nodes."""
        self.graph_storage.prune_low_importance_nodes(threshold)

    def iterative_refinement(self, reasoning_output: str) -> None:
        """Update the graph based on structured reasoning output."""
        try:
            chain_obj = parse_chain_of_thought(reasoning_output)
            self.reasoning_engine.update_from_chain_of_thought(chain_obj)
        except Exception as e:
            console.log(
                f"[Error] Failed to parse reasoning output: {e}", style="warning"
            )


def generate_with_context(
    query: str,
    context_manager: ContextGraphManager,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Constructs an extended reasoning prompt where the LLM explicitly navigates the graph.
    """
    # Get relevant nodes
    relevant_nodes = context_manager.query_context(query, top_k=3)

    # Build navigation path
    navigation_path = " → ".join(relevant_nodes)  # e.g., "Caching → Eviction → LRU"

    # Construct prompt
    extended_prompt = f"""
[Knowledge Graph Navigation]
- Your goal is to expand relevant concepts based on the **existing graph**.
- Navigate down from the **root concept** to related subtopics.
- Prioritize depth over breadth—go deeper before adding new high-level topics.

[Current Query]: {query}
[Current Navigation Path]: {navigation_path}

[Existing Graph Structure]:
{context_manager.visualize_graph_as_text()} 

[Reasoning Instructions]:
1️⃣ Identify missing knowledge gaps in the structure.
2️⃣ Expand deeper where necessary—**do not just add random new nodes.**
3️⃣ Maintain a logical structure using causality and dependencies.
4️⃣ Output only **valid JSON inside <json>...</json>** tags.

[Generated Knowledge Graph Update]:
"""

    console.log(f"Prompt: {extended_prompt}", style="prompt")

    # Tokenize and generate
    inputs = context_manager.tokenizer(extended_prompt, return_tensors="pt").to(
        context_manager.model.device
    )

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=context_manager.tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output = context_manager.model.generate(
            **inputs, generation_config=generation_config
        )

    # Decode and return
    return context_manager.tokenizer.decode(output[0], skip_special_tokens=True)


def get_context_mgr(model_name: str = MODEL_NAME) -> ContextGraphManager:
    """Create and initialize a context graph manager."""
    model = get_llm_model(model_name=model_name)
    tokenizer = get_tokenizer(model_name=model_name)
    return ContextGraphManager(tokenizer=tokenizer, model=model)


def seed_nodes(context_manager: ContextGraphManager, seed_data: list[SeedData]) -> None:
    """Add seed data to the context graph."""
    for data in seed_data:
        context_manager.add_context(data.node_id, data.content, data.metadata)


def chat_entry(
    context_manager: ContextGraphManager, user_input: str, conversation_turn: int
) -> None:
    """Process a single conversation turn."""
    # Decay node importance
    context_manager.decay_importance(
        decay_factor=IMPORTANCE_DECAY_FACTOR, adaptive=True
    )

    # Log user input
    console.log(f"[User {conversation_turn}]: {user_input}", style="user")

    # Add user input to context
    context_manager.add_context(f"user_{conversation_turn}", user_input)

    # Retrieve relevant context
    retrieved_context = context_manager.query_context(user_input, top_k=3)
    console.log(f"[Context Retrieved]: {retrieved_context}", style="context")

    # Generate response
    response = generate_with_context(user_input, context_manager=context_manager)
    console.log(f"[LLM Response {conversation_turn}]: {response}", style="llm")

    # Add response to context
    context_manager.add_context(f"llm_{conversation_turn}", response)

    # Extract reasoning and update graph
    try:
        reasoning_output = extract_and_clean_json(response)
        context_manager.iterative_refinement(reasoning_output)
    except Exception as e:
        console.log(
            f"[Error] No valid structured reasoning found: {e}. Raw LLM output: {response}",
            style="warning",
        )

    # Prune low-importance nodes
    context_manager.prune_context(threshold=PRUNE_THRESHOLD)


def simulate_chat(
    context_manager: ContextGraphManager,
    conversation_inputs: List[str],
    seed_data: Optional[List[SeedData]] = None,
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
