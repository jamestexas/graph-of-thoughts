import logging
import time

import torch
from main import (
    LLM_PATH,
    add_chain_of_thought_to_graph,
    build_structured_prompt,
    get_context_mgr,
    parse_chain_of_thought,
    update_and_save_graph,
)
from transformers import GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Stress test queries
test_queries = [
    "How can we optimize our caching mechanism to improve performance?",
    "What are the best strategies to handle high database read loads?",
    "How can we distribute our cache across multiple data centers?",
    "What are the benefits and downsides of implementing a write-through cache?",
    "If our system grows to 500M users, what caching strategy adjustments should we make?",
    "Can AI be used to dynamically optimize cache eviction policies?",
    "Our microservices rely heavily on caching. How can we ensure consistency across services?",
    "What techniques improve cache hit rates while minimizing memory usage?",
    "We have unpredictable traffic spikes. What caching strategies can handle sudden load increases?",
    "Compare LRU, LFU, and ARC caching policies for a real-time streaming service.",
    "What happens if we introduce an eventual consistency model into our caching layer?",
    "How do distributed databases optimize caching at the query layer?",
]


def run_stress_test():
    """
    Runs a stress test by sending multiple structured reasoning prompts to the system
    and validating the responses.
    """
    context_manager = get_context_mgr()
    failures = 0
    total_time = 0

    logging.info(f"🚀 Starting stress test with {len(test_queries)} queries.")

    for i, query in enumerate(test_queries):
        logging.info(f"\n[Test {i + 1}] Query: {query}")

        # Construct structured prompt
        structured_prompt = build_structured_prompt(query)

        # Measure time taken for inference
        start_time = time.time()

        try:
            # Tokenize and generate response
            inputs = context_manager.tokenizer(
                structured_prompt, return_tensors="pt"
            ).to(context_manager.model.device)

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

            if output is None or output.numel() == 0:
                logging.error("❌ Model output is empty. Skipping this query.")
                failures += 1
                continue

            structured_reasoning_output = context_manager.tokenizer.decode(
                output[0], skip_special_tokens=True
            )

            if not structured_reasoning_output.strip():
                logging.error("❌ Decoded LLM output is empty!")
                failures += 1
                continue

            logging.debug(f"🔍 Raw LLM Output:\n{structured_reasoning_output}")

            # Try parsing JSON output
            try:
                chain_obj = parse_chain_of_thought(structured_reasoning_output)
            except Exception as e:
                logging.error(f"❌ Failed to parse JSON output: {e}")
                logging.debug(f"🚨 Raw LLM Output: {structured_reasoning_output}")
                failures += 1
                continue

            if not hasattr(chain_obj, "nodes") or not hasattr(chain_obj, "edges"):
                logging.error("❌ Invalid ChainOfThought object detected.")
                failures += 1
                continue

            node_keys = {k.lower() for k in chain_obj.nodes.keys()}
            for edge in chain_obj.edges:
                if not all(node.lower() in node_keys for node in edge):
                    logging.error(
                        f"❌ Edge {edge} references missing node! Nodes available: {node_keys}"
                    )
                    failures += 1
                    continue

            # Add the reasoning chain to the context graph
            logging.debug(f"Graph before update: {context_manager.graph}")
            add_chain_of_thought_to_graph(chain_obj, context_manager)
            logging.debug(f"Graph after update: {context_manager.graph}")

            # 🔹 Save updated graph to disk (now merges instead of overwriting)
            update_and_save_graph(
                context_manager, LLM_PATH, structured_reasoning_output
            )

            logging.info("✅ Passed!")

        except Exception as e:
            logging.error(f"❌ Failed due to error: {str(e)}")

        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        logging.info(f"⏱️ Response Time: {elapsed_time:.2f} sec")

    logging.info("\n🎯 Stress Test Complete 🎯")
    logging.info(f"Total Tests: {len(test_queries)}, Failures: {failures}")
    logging.info(
        f"Total Execution Time: {total_time:.2f} sec, Avg Time per Query: {total_time / len(test_queries):.2f} sec"
    )


if __name__ == "__main__":
    run_stress_test()
