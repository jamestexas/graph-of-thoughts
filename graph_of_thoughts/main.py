# main.py

import json
import os

import torch
from transformers import GenerationConfig

from graph_of_thoughts.constants import LLM_PATH, OUTPUT_DIR
from graph_of_thoughts.context_manager import (
    console,
    get_context_mgr,
    parse_chain_of_thought,
    simulate_chat,
)
from graph_of_thoughts.models import SeedData
from graph_of_thoughts.utils import build_structured_prompt
from graph_of_thoughts.graph_utils import load_graph


# Define the output directory and file path for the LLM-generated graph

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        output = context_manager.model.generate(
            **inputs, generation_config=generation_config
        )

    structured_reasoning_output = context_manager.tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    console.log(
        "\n[Structured Reasoning Output]:", structured_reasoning_output, style="llm"
    )

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
        console.log(
            f"[Error] Could not parse chain-of-thought JSON: {e}", style="warning"
        )
        console.log(f"[Error] Raw LLM reasoning output: {output[0]}", style="warning")


if __name__ == "__main__":
    main()
