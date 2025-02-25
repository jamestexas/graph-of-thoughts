#!/usr/bin/env python
"""
POC: LLM Self-Managed Context Using an In-Memory Graph (NetworkX)
with Rich-Styled Logging

This script demonstrates a proof-of-concept where an LLM (GPT-2)
manages its own context in a dynamic graph. The graph (built with NetworkX)
stores context nodes (e.g., notes, file snippets, goals) along with metadata.
When a new query is received, the system retrieves the most relevant nodes,
appends their summarized contents to the prompt, and then generates output.
Rich is used to stylize printed output for easier debugging.
"""

from pydantic import BaseModel

import time
import torch
import networkx as nx
import numpy as np
import re
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.theme import Theme

# Define a custom theme for styled logging
custom_theme = Theme({
    "context": "cyan",
    "user": "magenta",
    "llm": "green",
    "prompt": "blue",
    "metrics": "yellow",
    "info": "white",
    "warning": "red bold",
})
console = Console(theme=custom_theme)

# Device selection (use MPS if available on Mac, else CPU)
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 100


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def summarize_text(text: str, max_sentences: int = 1) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences])


def deduplicate_context(
    context_texts: List[str],
    sentence_model: SentenceTransformer,
    threshold: float = 0.9,
) -> List[str]:
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


class ContextGraphManager:
    def __init__(self, tokenizer, model):
        self.graph = nx.DiGraph()
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_context(self, node_id: str, content: str, metadata: dict = None) -> None:
        if metadata is None:
            metadata = {}
        metadata.setdefault("importance", 1.0)
        self.graph.add_node(node_id, content=content, metadata=metadata)
        console.print(
            f"[ContextGraph] Added node '{node_id}' with importance {metadata['importance']}.",
            style="context",
        )

    def _compute_embedding(self, text: str) -> np.ndarray:
        embedding = self.sentence_model.encode(text, convert_to_numpy=True)
        return embedding.reshape(1, -1)

    def query_context(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self._compute_embedding(query)
        similarities = []
        for node_id, data in self.graph.nodes(data=True):
            content = data.get("content", "")
            if content.strip().lower() == query.strip().lower():
                continue
            node_emb = self._compute_embedding(content)
            sim = cosine_similarity(query_emb, node_emb)[0][0]
            if sim < 0.3:
                continue
            similarities.append((node_id, sim))
            console.print(
                f"[ContextGraph] Similarity for node '{node_id}': {sim:.4f}",
                style="context",
            )
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [
            self.graph.nodes[node_id]["content"] for node_id, _ in similarities[:top_k]
        ]
        for node_id, sim in similarities[:top_k]:
            self.graph.nodes[node_id]["metadata"]["importance"] += sim * 0.1
        if similarities:
            avg_sim = sum(sim for _, sim in similarities[:top_k]) / len(
                similarities[:top_k]
            )
            console.print(
                f"[Metrics] Average similarity of retrieved nodes: {avg_sim:.4f}",
                style="metrics",
            )
        return top_nodes

    def decay_importance(self, decay_factor: float = 0.95) -> None:
        for node_id in self.graph.nodes():
            self.graph.nodes[node_id]["metadata"]["importance"] *= decay_factor

    def prune_context(self, threshold: float = 0.8) -> None:
        nodes_to_remove = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("metadata", {}).get("importance", 1) < threshold:
                nodes_to_remove.append(node_id)
        for node_id in nodes_to_remove:
            self.graph.remove_node(node_id)
            console.print(
                f"[ContextGraph] Pruned node '{node_id}' due to low importance.",
                style="warning",
            )


def generate_with_context(
    query: str, context_texts: List[str], context_manager: ContextGraphManager
) -> str:
    summarized_contexts = [f"[Note]: {summarize_text(text)}" for text in context_texts]
    unique_contexts = deduplicate_context(
        summarized_contexts, context_manager.sentence_model
    )
    context_block = "\n".join(unique_contexts)
    extended_prompt = (
        f"{context_block}\n\n"
        f"[Query]: {query}\n"
        "[Instruction]: Please provide an answer that takes into account the above notes. "
        "Explain your reasoning and reference how each note influences your answer explicitly.\n"
        "[Answer]:"
    )
    extended_prompt = trim_prompt(
        extended_prompt, context_manager.tokenizer, max_tokens=800
    )
    console.print("\n[Extended Prompt]:", extended_prompt, style="prompt")
    inputs = context_manager.tokenizer(extended_prompt, return_tensors="pt").to(
        context_manager.model.device
    )
    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=context_manager.tokenizer.eos_token_id,
    )
    start_time = time.time()
    with torch.no_grad():
        output = context_manager.model.generate(
            **inputs, generation_config=generation_config
        )
    response_time = time.time() - start_time
    generated_text = context_manager.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    prompt_tokens = len(context_manager.tokenizer.encode(extended_prompt))
    console.print(
        f"[Metrics] Extended prompt token count: {prompt_tokens}", style="metrics"
    )
    console.print(
        f"[Metrics] Response generation time: {response_time:.2f} seconds",
        style="metrics",
    )
    return clean_text(generated_text)


def get_context_mgr() -> ContextGraphManager:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, return_dict_in_generate=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return ContextGraphManager(tokenizer=tokenizer, model=model)


class SeedData(BaseModel):
    node_id: str
    content: str
    metadata: dict


def seed_nodes(context_manager: ContextGraphManager, seed_data: List[SeedData]) -> None:
    for data in seed_data:
        context_manager.add_context(data.node_id, data.content, data.metadata)


def chat_entry(
    context_manager: ContextGraphManager, user_input: str, conversation_turn: int
) -> None:
    context_manager.decay_importance()
    console.print(f"[User {conversation_turn}]: {user_input}", style="user")
    context_manager.add_context(f"user_{conversation_turn}", user_input)
    retrieved_context = context_manager.query_context(user_input, top_k=3)
    response = generate_with_context(
        user_input, retrieved_context, context_manager=context_manager
    )
    console.print(f"[LLM Response {conversation_turn}]: {response}", style="llm")
    context_manager.add_context(f"llm_{conversation_turn}", response)
    context_manager.prune_context(threshold=0.8)


def simulate_chat(
    context_manager: ContextGraphManager,
    conversation_inputs: List[str],
    seed_data: List[SeedData] | None = None,
) -> None:
    seed_data = seed_data or [
        SeedData(
            node_id="node1",
            content="The repo includes a file that implements a caching mechanism for faster database queries.",
            metadata=dict(importance=0.9),
        ),
        SeedData(
            node_id="node2",
            content="The project goal is to improve inference time by dynamically managing context without overwhelming the prompt.",
            metadata=dict(importance=1.0),
        ),
    ]
    seed_nodes(context_manager, seed_data)
    conversation_turn = 1
    for user_input in conversation_inputs:
        console.print(f"\n[User {conversation_turn}]: {user_input}", style="user")
        chat_entry(
            context_manager=context_manager,
            user_input=user_input,
            conversation_turn=conversation_turn,
        )
        conversation_turn += 1


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
    # For interactive chat, uncomment the next line:
    # chat_loop(context_manager)
