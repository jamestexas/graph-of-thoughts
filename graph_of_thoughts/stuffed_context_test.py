from context_manager import ContextGraphManager, generate_with_context
from main import update_and_save_graph
import logging
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Debugging
print(f"Current working directory: {os.getcwd()}")
if os.path.exists("files"):
    print(f"Files directory contents: {os.listdir('files')}")
else:
    print("Files directory does not exist.")


def load_stuffed_context(filepath: str):
    """Loads and parses a stuffed context file."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    turns = []
    for i in range(0, len(lines), 2):
        user_line = lines[i].strip().replace("User: ", "")
        llm_line = lines[i + 1].strip().replace("LLM: ", "")
        turns.append({"user": user_line, "llm": llm_line})
    return turns


def run_stuffed_context_test(context_filepath: str, final_query: str, output_dir: str):
    """Runs a test with a stuffed context and a final query."""

    context_manager = ContextGraphManager()

    stuffed_turns = load_stuffed_context(context_filepath)

    # initialize the graph with the stuffed context.
    turn_num = 1
    for turn in stuffed_turns:
        context_manager.add_context(f"user_{turn_num}", turn["user"])
        context_manager.add_context(f"llm_{turn_num}", turn["llm"])
        turn_num += 1

    # Run the final query
    logging.info(f"Final Query: {final_query}")

    # Use your existing chat_entry or generate_with_context function here
    response = generate_with_context(final_query, context_manager)
    logging.info(f"LLM Response: {response}")

    try:
        reasoning_output = context_manager.extract_and_clean_json(response)
        context_manager.iterative_refinement(reasoning_output)
        # make sure the output dir exists.
        os.makedirs(output_dir, exist_ok=True)
        update_and_save_graph(
            context_manager, os.path.join(output_dir, "llm_graph.json"), response
        )

        # save the experiment data.
        experiment_data = {
            "stuffed_context": load_stuffed_context(context_filepath),
            "final_query": final_query,
            "llm_response": response,
            "graph_after": context_manager.graph_to_json(),
        }

        with open(
            os.path.join(output_dir, "stuffed_context_experiment_data.json"), "w"
        ) as f:
            json.dump(experiment_data, f, indent=4)

    except Exception as e:
        logging.error(
            f"[Error] No valid structured reasoning found: {e}. Raw LLM output: {response}",
            style="warning",
        )


if __name__ == "__main__":
    # Create test stuffed context files in a "files" directory.
    # Create an "output_stuffed" directory to save the results.
    run_stuffed_context_test(
        "files/stuffed_context_1.txt",
        "Can we optimize the cache eviction policy for better performance?",
        "output_stuffed/test1",
    )
    run_stuffed_context_test(
        "files/stuffed_context_2.txt",
        "What are the security implications of caching sensitive data?",
        "output_stuffed/test2",
    )
    run_stuffed_context_test(
        "files/stuffed_context_3.txt",
        "Compare and contrast Redis and Memcached.",
        "output_stuffed/test3",
    )
