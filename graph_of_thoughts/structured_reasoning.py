# structured_reasoning.py
from pydantic import ValidationError
from .context_manager import ContextGraphManager, console, parse_chain_of_thought
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def update_context_with_reasoning(
    reasoning_output: str, context_manager: ContextGraphManager
) -> None:
    """
    Parse the structured reasoning output from the LLM as valid JSON with "nodes" and "edges".
    If valid, add them to the context graph.
    """
    try:
        chain_obj = parse_chain_of_thought(reasoning_output)
        logging.debug(f"🔎 Parsed Chain of Thought Output: {chain_obj}")

    except (ValueError, ValidationError) as e:
        console.log(
            f"[Error] Failed to parse chain-of-thought JSON: {e}", style="warning"
        )
        return

    if not chain_obj.nodes:
        console.log("[Error] The chain-of-thought has no 'nodes'.", style="warning")
        return

    for node_id, description in chain_obj.nodes.items():
        new_node_id = f"reason_{node_id}"
        context_manager.add_context(new_node_id, description)
    for source, target in chain_obj.edges:
        console.log(f"[Structured Edge] {source} -> {target}", style="info")
