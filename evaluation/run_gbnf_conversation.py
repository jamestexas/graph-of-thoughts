"""
run_gbnf_conversation.py

A demo conversation script that uses llama_cpp as the inference backend with a GBNF grammar.
This script loads a GGUF model via llama_cpp (with grammar enforcement), plugs it into your
Graph of Thoughts infrastructure, and runs a multi-turn conversation.
"""

import time
from pathlib import Path

# Import llama_cpp (make sure you have installed llama-cpp-python)
from llama_cpp import Llama

from graph_of_thoughts.chat_manager import ChatManager
from graph_of_thoughts.constants import console
from graph_of_thoughts.context_manager import get_context_mgr
from graph_of_thoughts.utils import get_unified_llm_model

# --- Configuration ---

MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/a5594fb18df5dfc6b43281423fcce6750cd92de5/Llama-3.2-1B-Instruct-Q8_0.gguf"
).expanduser()
GRAMMAR_FILE = "chain_of_thought.gbnf"

# Create a simple GBNF grammar specification for our ChainOfThought output.
# This example forces output with exactly two keys: "nodes" and "edges".
GRAMMAR_SPEC = r"""
# GBNF grammar for ChainOfThought
root ::= "{" ws "\"nodes\"" ws ":" ws nodes ws "," ws "\"edges\"" ws ":" ws edges ws "}"
nodes ::= "{" ws node_pairs ws "}"
node_pairs ::= node_pair (ws "," ws node_pair)*
node_pair ::= string ws ":" ws string
edges ::= "[" ws edge_list ws "]"
edge_list ::= edge (ws "," ws edge)*
edge ::= "[" ws string ws "," ws string ws "]"
string ::= "\"" [^\"]+ "\""
ws ::= [ \t\n]*
"""


# Write the grammar spec to a file if it doesn't exist.
def get_grammar_path(grammar_file: str = GRAMMAR_FILE) -> Path:
    """Get the Path object for the grammar file."""
    grammar_path = Path(grammar_file)
    if not grammar_path.exists():
        grammar_path.write_text(GRAMMAR_SPEC)
        console.log(f"Grammar file written to {grammar_path}", style="info")
    return grammar_path


def run_conversation():
    """Run a multi-turn conversation using ChatManager with grammar enforcement."""
    # --- Initialize the llama_cpp model ---
    console.log("Loading GGUF model via llama_cpp...", style="info")
    llama_model = Llama(
        model_path=str(MODEL_PATH),
        # Pass the grammar file to enforce output constraints:
        grammar_file=str(get_grammar_path()),
        n_ctx=2048,
        seed=42,
        verbose=True,
    )

    # Create a unified model using the llama_cpp backend
    unified_model = get_unified_llm_model(
        backend="llama_cpp",
        model=llama_model,
    )

    # Now create a ContextGraphManager using the get_context_mgr helper
    context_manager = get_context_mgr(unified_model=unified_model)

    # --- Instantiate ChatManager with the context manager ---
    chat_manager = ChatManager(context_manager=context_manager)

    # Define the conversation turns
    conversation_turns = [
        "Tell me about database indexing and why it's useful.",
        "What are the different types of database indices?",
        "When would you choose a B-tree over a hash index?",
    ]

    console.log("=== Starting GGUF Conversation with GBNF Grammar ===", style="bold green")

    for turn_idx, turn in enumerate(conversation_turns, start=1):
        response = chat_manager.process_turn(turn, turn_idx)
        console.log(f"[LLM Response {turn_idx}]: {response}", style="llm")
        time.sleep(1)  # small pause for clarity

    # Display the final knowledge graph as text
    console.log("\n[Final Context Graph]:", style="context")
    graph_text = context_manager.visualize_graph_as_text()
    console.log(graph_text, style="context")


if __name__ == "__main__":
    run_conversation()
