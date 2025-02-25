# main.py

import json
import os
import shutil
from pathlib import Path

import torch
from transformers import GenerationConfig

from graph_of_thoughts.constants import LLM_PATH, OUTPUT_DIR
from graph_of_thoughts.context_manager import (
    ContextGraphManager,
    console,
    get_context_mgr,
    parse_chain_of_thought,
    simulate_chat,
)
from graph_of_thoughts.models import ChainOfThought, SeedData
from graph_of_thoughts.utils import build_structured_prompt

# Define the output directory and file path for the LLM-generated graph

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_chain_of_thought_to_graph(
    chain_obj: ChainOfThought,
    context_manager: ContextGraphManager,
) -> None:
    """
    Adds nodes and edges from the parsed ChainOfThought to the context graph.
    """
    for node_id, desc in chain_obj.nodes.items():
        new_node_id = f"reason_{node_id}"  # Unique prefix to prevent conflicts
        context_manager.add_context(new_node_id, desc)

    for source, target in chain_obj.edges:
        console.log(f"[Structured Edge] {source} -> {target}", style="info")


##############################################
# 4) Main: Combine normal Q&A + structured reasoning
##############################################


def update_and_save_graph(context_manager, output_path: str, structured_reasoning_output: str):
    """
    Load the existing LLM graph, update it with the new structured reasoning output,
    and save it back to file.
    """

    # 🔹 Save raw LLM output for debugging
    debug_output_path = "debug_llm_output.txt"
    with open(debug_output_path, "w") as f:
        f.write(structured_reasoning_output)
    console.log(f"📝 LLM output saved to {debug_output_path} for inspection.")

    # 🔹 Ensure structured output isn't empty
    if not structured_reasoning_output.strip():
        console.log("❌ LLM output is empty. Skipping update.")
        return

    # 🔹 Load existing graph if available
    if os.path.exists(output_path) and os.path.getsize(output_path) > 2:  # Ensure valid file
        with open(output_path) as f:
            try:
                existing_graph = json.load(f)
            except json.JSONDecodeError:
                console.log(f"⚠️ Corrupted JSON detected in {output_path}. Resetting graph.")
                existing_graph = {"nodes": {}, "edges": []}
    else:
        existing_graph = {"nodes": {}, "edges": []}

    # 🔹 Parse structured reasoning output
    try:
        chain_obj = parse_chain_of_thought(structured_reasoning_output)
        if not chain_obj.nodes or not chain_obj.edges:
            console.log("❌ Parsed chain of thought has missing nodes or edges.")
            return
    except ValueError as e:
        console.log(f"❌ Error parsing structured reasoning: {e}")
        console.log(f"🚨 Raw Output:\n{structured_reasoning_output}")
        return

    # 🔹 Merge new nodes & edges (instead of overwriting)
    for node, desc in chain_obj.nodes.items():
        if node not in existing_graph["nodes"]:  # Prevent overwriting existing descriptions
            existing_graph["nodes"][node] = desc

    for edge in chain_obj.edges:
        if edge not in existing_graph["edges"]:  # Avoid duplicate edges
            existing_graph["edges"].append(edge)

    # 🔹 Safely write updated graph to a temp file first
    temp_path = output_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(existing_graph, f, indent=2)

    shutil.move(temp_path, output_path)  # Atomic move to prevent corruption
    console.log(f"✅ Updated LLM Graph saved to {output_path}")


def load_graph(file_path: str = LLM_PATH) -> dict:
    """Load graph data from a JSON file."""
    p = Path(file_path).resolve()
    default_graph = dict(nodes=dict(), edges=[])
    if not p.exists():
        return default_graph
    contents = p.read_text()
    return json.loads(contents)


def main():
    context_manager = get_context_mgr()
    # Load previous LLM graph if available
    llm_graph = load_graph(LLM_PATH)
    # Seed nodes for initial context
    seed_data = [
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

    # 1️⃣ Simulate conversation to refine context
    canned_conversation = [
        "How can I improve the caching mechanism without increasing latency?",
        "What strategies can reduce database load during peak hours?",
    ]
    simulate_chat(
        context_manager=context_manager,
        conversation_inputs=canned_conversation,
        seed_data=seed_data,
    )

    # 2️⃣ Ask a structured reasoning question
    question = "How can we optimize our caching mechanism to improve performance?"
    structured_prompt = build_structured_prompt(question)

    console.log("\n[Structured Reasoning Prompt]:", structured_prompt, style="prompt")

    # 3️⃣ Run inference with LLM
    inputs = context_manager.tokenizer(structured_prompt, return_tensors="pt").to(
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
        output = context_manager.model.generate(**inputs, generation_config=generation_config)

    structured_reasoning_output = context_manager.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    console.log("\n[Structured Reasoning Output]:", structured_reasoning_output, style="llm")

    try:
        chain_obj = parse_chain_of_thought(structured_reasoning_output)

        # 4️⃣ Merge with the existing knowledge graph
        llm_graph["nodes"].update(chain_obj.nodes)  # Add new nodes

        for edge in chain_obj.edges:
            if edge not in llm_graph["edges"]:  # Avoid duplicate edges
                llm_graph["edges"].append(edge)

        console.log("[Info] Merging structured reasoning into graph...", style="info")

        # 5️⃣ Save updated graph
        with open(LLM_PATH, "w") as f:
            json.dump(llm_graph, f, indent=2)
        console.log(f"✅ Updated LLM Graph saved to {LLM_PATH}", style="info")

    except Exception as e:
        console.log(f"[Error] Could not parse chain-of-thought JSON: {e}", style="warning")
        console.log(f"[Error] Raw LLM reasoning output: {output[0]}", style="warning")


if __name__ == "__main__":
    main()
