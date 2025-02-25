# # src/context_keep/expirement/dynamic_context/context_manager.py

import json
import os
from pydantic import BaseModel, Field, field_validator
import torch
import networkx as nx
import numpy as np
import re
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig
from rich.console import Console
from rich.theme import Theme
from datetime import datetime, timezone
import faiss
from evaluate_llm_graph import load_graph, compute_graph_metrics, KnowledgeGraph


from transformers import AutoTokenizer, AutoModelForCausalLM

import logging

logging.basicConfig(level=logging.INFO)  # Enable debug logs

# Define a custom theme for styled logging
custom_theme = Theme(
    dict(
        context="cyan",
        user="magenta",
        llm="green",
        prompt="blue",
        metrics="yellow",
        info="white",
        warning="bold red",
    )
)
console = Console(theme=custom_theme)
MAX_NEW_TOKENS = 100
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"  # default to maccy


class FastEmbeddingIndex:
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


# 1) Define a Pydantic model that enforces "nodes" and "edges" structure
class ChainOfThought(BaseModel):
    """
    Represents the JSON structure we want:
        {
            "nodes": { "T1": "desc", "T2": "desc" },
            "edges": [ ["T2","T1"], ... ]
        }
    """

    nodes: dict[str, str]
    edges: list[list[str]]

    @field_validator("edges")
    def check_edges_not_empty(cls, edges):
        for e in edges:
            if len(e) != 2:
                raise ValueError(f"Edge must have exactly 2 items, got: {e}")
            if not e[0] or not e[1]:
                raise ValueError(f"Edge has empty source/target: {e}")
        return edges


def extract_json_substring(raw_output: str):
    """
    Extracts and validates JSON from an LLM response while logging issues.
    """
    logging.debug(f"🔍 Raw LLM Output:\n{raw_output}")

    # Normalize inconsistent JSON markers
    raw_output = raw_output.replace("$json</json>", "").replace("$json", "").strip()

    # First try: Extract properly wrapped JSON
    json_tags_regex = re.compile(r"<json>\s*(\{.*?\})\s*</json>", re.DOTALL)
    if not (match := json_tags_regex.search(raw_output)):
        logging.warning("⚠️ No <json>...</json> tags found. Trying direct extraction...")
        match = re.search(r"(\{.*?\})", raw_output, re.DOTALL)

    if not match:
        logging.error("❌ No valid JSON found in output.")
        return None

    json_str = match.group(1).strip()
    logging.debug(f"✅ Extracted JSON Candidate:\n{json_str}")

    # Ensure JSON contains required fields
    try:
        parsed_json = json.loads(json_str)

        # Check if required fields exist
        if "nodes" not in parsed_json or "edges" not in parsed_json:
            logging.error(f"❌ Missing required fields in JSON: {parsed_json}")
            return None

        logging.info("✅ Successfully parsed and validated JSON!")
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parsing failed: {e}")
        return None


def parse_chain_of_thought(raw_output: str) -> ChainOfThought:
    """
    Extracts and validates structured JSON from LLM output, ensuring robustness.
    """

    # 🔍 Extract JSON inside <json>...</json>, allowing for any whitespace or newlines
    logging.debug(f"🔍 Extracting JSON from output:\n{raw_output}")

    match = re.search(r"<json>\s*(\{.*?\})\s*</json>", raw_output, re.DOTALL)

    if not match:
        logging.error("❌ No valid JSON block found in LLM output!")
        raise ValueError("No valid JSON block found.")

    json_string = match.group(1).strip()
    # ✅ Debugging: Print extracted JSON before parsing
    logging.debug(f"✅ Extracted JSON String:\n{json_string}")

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON Decode Error: {e}\nRaw JSON Extracted:\n{json_string}")
        raise ValueError("Invalid JSON format.")

    if "nodes" not in data or "edges" not in data:
        logging.error(f"❌ Parsed JSON missing required fields: {data}")
        raise ValueError("JSON missing 'nodes' or 'edges'.")

    return ChainOfThought(nodes=data["nodes"], edges=data["edges"])


def summarize_text(text: str, max_sentences: int = 1) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences])


def extract_balanced_json(text: str) -> str:
    """
    Extracts the first balanced JSON object from the given text.

    Args:
        text (str): The text containing a JSON block.

    Returns:
        str: The extracted JSON string.

    Raises:
        ValueError: If no balanced JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the text.")
    stack = []
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            stack.append("{")
        elif char == "}":
            stack.pop()
            if not stack:
                candidate = text[start : i + 1]
                try:
                    # Re-serialize to ensure it's clean
                    parsed = json.loads(candidate)
                    return json.dumps(parsed)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Extracted JSON is invalid: {e}")
    raise ValueError("Unbalanced JSON braces in the text.")


def extract_and_clean_json(text: str) -> str:
    """
    Extracts the first balanced JSON object from the text and cleans it by re-serializing.

    Args:
        text (str): The text containing a JSON block.

    Returns:
        str: A clean JSON string.

    Raises:
        ValueError: If no valid JSON object is found or it is invalid.
    """
    raw_json = extract_balanced_json(text)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted JSON is invalid: {e}")
    return json.dumps(parsed)


class ContextNode(BaseModel):
    node_id: str = Field(
        description="A unique identifier for the context node.",
    )
    content: str = Field(
        description="The content of the context node.",
    )
    metadata: dict[str, float] = Field(
        default_factory=lambda: dict(importance=1.0),
        description="A dictionary of metadata associated with the context node.",
    )
    created_at: datetime = Field(
        description="The timestamp when the context node was created.",
        default_factory=lambda: datetime.now(timezone.utc),
    )


class SeedData(BaseModel):
    node_id: str
    content: str
    metadata: dict


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines."""
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_context(
    context_texts: List[str],
    sentence_model: SentenceTransformer,
    threshold: float = 0.85,
) -> List[str]:
    """
    Remove context entries that are nearly identical based on cosine similarity.
    """
    unique_contexts = []
    embeddings = []
    for text in context_texts:
        emb = sentence_model.encode(text, convert_to_numpy=True)
        duplicate = False
        for existing_emb in embeddings:
            if (
                cosine_similarity(emb.reshape(1, -1), existing_emb.reshape(1, -1))[0][0]
                > threshold
            ):
                duplicate = True
                break
        if not duplicate:
            unique_contexts.append(text)
            embeddings.append(emb)
    return unique_contexts


def trim_prompt(extended_prompt: str, tokenizer, max_tokens: int = 800) -> str:
    """
    Ensure the extended prompt does not exceed a maximum token count.
    """
    tokens = tokenizer.encode(extended_prompt)
    if len(tokens) <= max_tokens:
        return extended_prompt
    lines = extended_prompt.split("\n")
    trimmed_lines = []
    current_tokens = 0
    for line in lines:
        line_tokens = tokenizer.encode(line)
        if current_tokens + len(line_tokens) <= max_tokens:
            trimmed_lines.append(line)
            current_tokens += len(line_tokens)
        else:
            break
    return "\n".join(trimmed_lines)


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
    node_subA = ContextNode(
        node_id="subA",
        content="A sub concept under root",
        metadata={"importance": 0.9},
    )
    graph.add_node("subA", data=node_subA)

    # subB node
    node_subB = ContextNode(
        node_id="subB",
        content="Another sub concept under root",
        metadata={"importance": 0.9},
    )
    graph.add_node("subB", data=node_subB)

    # edges
    graph.add_edge("root", "subA")
    graph.add_edge("root", "subB")

    return graph


class ContextGraphManager:
    """
    Manages a dynamic context graph using NetworkX.
    Each node is a ContextNode (Pydantic model) stored under the attribute 'data'.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer | None = None,
        model: AutoModelForCausalLM | None = None,
        initial_graph: None | nx.DiGraph = None,
    ):
        if initial_graph is None:
            self.graph = build_initial_graph()
        else:
            self.graph = initial_graph.copy()
        self.tokenizer = tokenizer or get_tokenizer()
        self.model = model or get_llm_model()
        self.model.eval()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_index = FastEmbeddingIndex()

    def graph_to_json(self):
        """Converts the graph to a JSON-serializable format."""
        data = nx.node_link_data(self.graph)
        for node in data["nodes"]:
            for key, value in node.items():
                if hasattr(value, "model_dump"):
                    node[key] = value.model_dump()
        return data

    def visualize_graph_as_text(self) -> str:
        """
        Returns a simple text representation of the context graph
        showing nodes and edges.
        """
        text_representation = []
        text_representation.append("📌 **Current Knowledge Graph**\n")

        # 🔹 Nodes
        text_representation.append("🟢 **Nodes**:")
        for node, data in self.graph.nodes(data=True):
            text_representation.append(
                f"  - {node}: {data.get('content', 'No description')}"
            )

        # 🔹 Edges
        text_representation.append("\n🔗 **Edges**:")
        for source, target in self.graph.edges():
            text_representation.append(f"  - {source} → {target}")

        return "\n".join(text_representation)

    def add_context(self, node_id: str, content: str, metadata: dict = None) -> None:
        embedding = self.sentence_model.encode(content, convert_to_numpy=True)
        self.embedding_index.add_node(node_id, embedding)
        self.graph.add_node(node_id, data={"content": content, "importance": 1.0})

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute a semantic embedding using Sentence Transformers."""
        embedding = self.sentence_model.encode(text, convert_to_numpy=True)
        return embedding.reshape(1, -1)

    def query_context(self, query: str, top_k: int = 3):
        if not self.graph.nodes:  # added check.
            return []

        console.log(f"[Query] Top-{top_k} nodes for query '{query}':", style="context")
        query_emb = self.sentence_model.encode(query)
        _query_result = self.embedding_index.query(
            embedding=query_emb,
            top_k=top_k,
        )
        console.log(f"[Query Result] {_query_result}", style="context")
        indices, _ = self.embedding_index.query(query_emb, top_k)

        if not indices or not indices[0]:  # added check.
            return []

        return [self.nodes[i] for i in indices[0]]

    def _manual_query_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-k context nodes based on cosine similarity with the query.
        """
        query_emb = self._compute_embedding(query)
        similarities = []
        for node_id, attr in self.graph.nodes(data=True):
            node: ContextNode = attr["data"]
            content = node.content
            if content.strip().lower() == query.strip().lower():
                continue
            node_emb = self._compute_embedding(content)
            sim = cosine_similarity(query_emb, node_emb)[0][0]
            if sim < 0.3:
                continue
            similarities.append((node_id, sim))
            console.log(
                f"[ContextGraph] Similarity for node '{node_id}': {sim:.4f}",
                style="context",
            )
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [
            self.graph.nodes[node_id]["data"].content
            for node_id, _ in similarities[:top_k]
        ]
        for node_id, sim in similarities[:top_k]:
            node: ContextNode = self.graph.nodes[node_id]["data"]
            node.metadata["importance"] += sim * 0.1
            self.graph.nodes[node_id]["data"] = node
        if similarities:
            avg_sim = sum(sim for _, sim in similarities[:top_k]) / len(
                similarities[:top_k]
            )
            console.log(
                f"[Metrics] Average similarity of retrieved nodes: {avg_sim:.4f}",
                style="metrics",
            )
        return top_nodes

    def decay_importance(
        self,
        decay_factor: float = 0.95,
        adaptive: bool = True,
    ) -> None:
        """
        Applies adaptive decay to all nodes in the context graph.
        """
        for node_id, attr in self.graph.nodes(data=True):
            node_data = attr.get("data", None)

            if node_data is None:
                continue  # Skip nodes without valid data

            if isinstance(node_data, dict):
                # Ensure the required 'node_id' field is included
                node_data["node_id"] = node_id
                node = ContextNode(**node_data)
            else:
                node = node_data  # Already a ContextNode

            # Apply adaptive decay
            new_importance = (
                self.adaptive_decay(node, decay_factor)
                if adaptive
                else node.metadata["importance"] * decay_factor
            )
            node.metadata["importance"] = new_importance

            # Store updated node back in the graph
            self.graph.nodes[node_id]["data"] = node

    def iterative_refinement(self, reasoning_output: str) -> None:
        """
        Parse the structured chain-of-thought JSON output and update the context graph.
        """
        try:
            chain_obj = parse_chain_of_thought(reasoning_output)
        except Exception as e:
            console.log(
                f"[Error] Failed to parse reasoning output: {e}", style="warning"
            )
            return

        for node_id, description in chain_obj.nodes.items():
            new_node_id = f"reason_{node_id}"
            self.add_context(new_node_id, description, metadata={"importance": 1.0})
        for source, target in chain_obj.edges:
            console.log(f"[Structured Edge] {source} -> {target}", style="info")
        console.log(
            "[Iterative Refinement] Context graph updated from reasoning output.",
            style="context",
        )

    def adaptive_decay(
        self,
        node: ContextNode | dict,
        decay_factor: float = 0.95,
    ) -> float:
        """
        Applies adaptive decay based on node age.

        Args:
            node (ContextNode | dict): The node whose importance should be decayed.
            decay_factor (float): The decay rate.

        Returns:
            float: The new importance score.
        """
        if isinstance(node, dict):
            if "node_id" not in node:
                raise ValueError("Missing node_id in node data during adaptive decay.")

            node = ContextNode(**node)  # Convert dict to ContextNode

        age = (datetime.now(timezone.utc) - node.created_at).total_seconds()
        decay_rate = decay_factor ** (age / 3600)  # Adjust decay over time (per hour)

        return node.metadata["importance"] * decay_rate

    def prune_context(self, threshold: float = 0.8) -> None:
        """
        Prunes low-importance nodes from the context.

        Args:
            threshold (float): The importance score below which nodes should be removed.
        """
        for node_id, attr in list(self.graph.nodes(data=True)):
            node_data = attr.get("data", None)

            if not node_data:
                continue

            # Ensure node is a ContextNode instance
            if isinstance(node_data, dict):
                node_data["node_id"] = node_id  # Ensure node_id is present
                node = ContextNode(**node_data)
            else:
                node = node_data  # Already a ContextNode

            if self.adaptive_decay(node) < threshold:
                self.graph.remove_node(node_id)


def generate_with_context(query: str, context_manager: ContextGraphManager) -> str:
    """
    Constructs an extended reasoning prompt where the LLM explicitly navigates the graph.
    """

    # 🔹 1. Identify relevant nodes
    relevant_nodes = context_manager.query_context(query, top_k=3)

    # 🔹 2. Construct hierarchical navigation prompt
    navigation_path = " → ".join(relevant_nodes)  # e.g., "Caching → Eviction → LRU"

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

    logging.debug(f"Prompt: {extended_prompt}")

    # Tokenize and generate response
    inputs = context_manager.tokenizer(extended_prompt, return_tensors="pt").to(
        context_manager.model.device
    )

    generation_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=context_manager.tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output = context_manager.model.generate(
            **inputs, generation_config=generation_config
        )

    structured_reasoning_output = context_manager.tokenizer.decode(
        output[0], skip_special_tokens=True
    )

    logging.debug(f"Raw LLM Output: {structured_reasoning_output}")

    return structured_reasoning_output


def get_llm_model(model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name, return_dict_in_generate=True, torch_dtype=torch.float16
    ).to(DEVICE)


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_context_mgr(model_name: str = MODEL_NAME) -> ContextGraphManager:
    model = get_llm_model(model_name=model_name)
    tokenizer = get_tokenizer(model_name=model_name)
    return ContextGraphManager(tokenizer=tokenizer, model=model)


def seed_nodes(context_manager: ContextGraphManager, seed_data: List[SeedData]) -> None:
    for data in seed_data:
        context_manager.add_context(data.node_id, data.content, data.metadata)


def chat_entry(
    context_manager: ContextGraphManager, user_input: str, conversation_turn: int
) -> None:
    context_manager.decay_importance(decay_factor=0.95, adaptive=True)
    console.log(f"[User {conversation_turn}]: {user_input}", style="user")
    context_manager.add_context(f"user_{conversation_turn}", user_input)
    retrieved_context = context_manager.query_context(user_input, top_k=3)
    console.log(f"[Context Retrieved]: {retrieved_context}", style="context")
    console.log("Forcing empty context retrieval...", style="context")
    response = generate_with_context(user_input, context_manager=context_manager)
    console.log(f"[LLM Response {conversation_turn}]: {response}", style="llm")
    context_manager.add_context(f"llm_{conversation_turn}", response)
    # Use our robust extraction method
    try:
        reasoning_output = extract_and_clean_json(response)
        context_manager.iterative_refinement(reasoning_output)
    except Exception as e:
        console.log(
            f"[Error] No valid structured reasoning found: {e}. Raw LLM output: {response}",
            style="warning",
        )
    context_manager.prune_context(threshold=0.8)


def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)  # Convert set to list
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def simulate_chat(
    context_manager: ContextGraphManager,
    conversation_inputs: List[str],
    seed_data: List[SeedData] | None = None,
    experiment_name: str = "default_experiment",
) -> None:
    # ... (seed_nodes code) ...
    conversation_turn = 1
    experiment_data = []  # list to hold all conversation turns.
    for user_input in conversation_inputs:
        console.log(f"\n[User {conversation_turn}]: {user_input}", style="user")
        graph_before = context_manager.graph.copy()  # capture the graph before.
        chat_entry(
            context_manager=context_manager,
            user_input=user_input,
            conversation_turn=conversation_turn,
        )
        graph_after = context_manager.graph.copy()  # capture the graph after.
        retrieved_context = context_manager.query_context(user_input, top_k=3)

        # Convert ContextNode objects to dictionaries
        retrieved_context_serializable = [
            node.model_dump() if hasattr(node, "model_dump") else str(node)
            for node in retrieved_context
        ]

        # save the graphs to json, so that they can be loaded by the evaluate script.
        baseline_path = "output/baseline_graph.json"
        llm_path = "output/llm_graph.json"

        os.makedirs("output", exist_ok=True)  # make sure the output directory exists.

        # create a baseline graph to compare against. this is a stand in.
        baseline = KnowledgeGraph("Baseline")
        baseline.graph = graph_before
        baseline.save_graph_json(baseline_path)

        # create the llm graph.
        llm = KnowledgeGraph("LLM")
        llm.graph = graph_after
        llm.save_graph_json(llm_path)

        # calculate the metrics
        metrics = compute_graph_metrics(load_graph(baseline_path), load_graph(llm_path))

        def serialize_graph_data(graph):
            data = nx.node_link_data(graph)
            for node in data["nodes"]:
                for key, value in node.items():
                    if hasattr(value, "model_dump"):
                        node[key] = value.model_dump()
            return data

        experiment_data.append({
            "turn": conversation_turn,
            "user_input": user_input,
            "llm_response": context_manager.graph.nodes[f"llm_{conversation_turn}"][
                "data"
            ]["content"],
            "retrieved_context": retrieved_context_serializable,
            "graph_before": serialize_graph_data(graph_before),
            "graph_after": serialize_graph_data(graph_after),
            "metrics": metrics,
        })
        conversation_turn += 1
    with open(f"{experiment_name}_data.json", "w") as f:
        json.dump(experiment_data, f, indent=4, default=datetime_handler)


if __name__ == "__main__":
    context_manager = get_context_mgr()
    canned_conversation = [
        "How can I improve the caching mechanism without increasing latency?",
        "What strategies can reduce database load during peak hours?",
        "Can we optimize the cache eviction policy for better performance?",
        "What are some best practices for scaling the caching system?",
    ]
    simulate_chat(
        context_manager=context_manager, conversation_inputs=canned_conversation
    )
    console.log("\n[Final Context Graph]:", context_manager.graph, style="context")
    console.log("[Info] Experiment completed.", style="info")
