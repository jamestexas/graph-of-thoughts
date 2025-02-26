"""
Utility functions for working with knowledge graphs in the Graph of Thoughts project.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from graph_of_thoughts.constants import LLM_PATH, OUTPUT_DIR, console
from graph_of_thoughts.context_manager import ContextGraphManager, parse_chain_of_thought
from graph_of_thoughts.models import ChainOfThought

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_chain_of_thought_to_graph(
    chain_obj: ChainOfThought,
    context_manager: ContextGraphManager,
) -> None:
    """
    Adds nodes and edges from the parsed ChainOfThought to the context graph.

    Args:
        chain_obj: The ChainOfThought object with nodes and edges
        context_manager: The context manager to add nodes and edges to
    """
    for node_id, desc in chain_obj.nodes.items():
        new_node_id = f"reason_{node_id}"  # Unique prefix to prevent conflicts
        context_manager.add_context(new_node_id, desc, metadata={"importance": 1.0})

    for source, target in chain_obj.edges:
        # Convert to our prefixed node IDs
        source_id, target_id = f"reason_{source}", f"reason_{target}"
        context_manager.graph_storage.add_edge(source_id, target_id)
        console.log(f"[Structured Edge] {source} -> {target}", style="info")


def update_and_save_graph(
    context_manager: Optional[ContextGraphManager],
    output_path: str,
    structured_reasoning_output: str,
    debug_path: Optional[str] = None,
) -> bool:
    """
    Load the existing LLM graph, update it with the new structured reasoning output,
    and save it back to file.

    Args:
        context_manager: Optional context manager (for logging)
        output_path: Path to save the updated graph
        structured_reasoning_output: Raw LLM output containing JSON
        debug_path: Optional path to save raw output for debugging

    Returns:
        bool: True if update successful, False otherwise
    """
    # Save raw LLM output for debugging if path provided
    if debug_path:
        with open(debug_path, "w") as f:
            f.write(structured_reasoning_output)
        console.log(f"📝 LLM output saved to {debug_path} for inspection.")

    # Ensure structured output isn't empty
    if not structured_reasoning_output.strip():
        console.log("❌ LLM output is empty. Skipping update.")
        return False

    # Load existing graph if available
    if os.path.exists(output_path) and os.path.getsize(output_path) > 2:  # Ensure valid file
        with open(output_path) as f:
            try:
                existing_graph = json.load(f)
            except json.JSONDecodeError:
                console.log(f"⚠️ Corrupted JSON detected in {output_path}. Resetting graph.")
                existing_graph = {"nodes": {}, "edges": []}
    else:
        existing_graph = {"nodes": {}, "edges": []}

    # Parse structured reasoning output
    try:
        chain_obj = parse_chain_of_thought(structured_reasoning_output)
        if not chain_obj.nodes or not chain_obj.edges:
            console.log("❌ Parsed chain of thought has missing nodes or edges.")
            return False
    except ValueError as e:
        console.log(f"❌ Error parsing structured reasoning: {e}")
        console.log(f"🚨 Raw Output:\n{structured_reasoning_output}")
        return False

    # Merge new nodes & edges (instead of overwriting)
    for node, desc in chain_obj.nodes.items():
        if node not in existing_graph["nodes"]:  # Prevent overwriting existing descriptions
            existing_graph["nodes"][node] = desc

    for edge in chain_obj.edges:
        if edge not in existing_graph["edges"]:  # Avoid duplicate edges
            existing_graph["edges"].append(edge)

    # Safely write updated graph to a temp file first
    temp_path = output_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(existing_graph, f, indent=2)

    shutil.move(temp_path, output_path)  # Atomic move to prevent corruption
    console.log(f"✅ Updated LLM Graph saved to {output_path}")
    return True


def load_graph(file_path: str = LLM_PATH) -> dict:
    """
    Load graph data from a JSON file.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        dict: The loaded graph data or a default empty graph if file doesn't exist
    """
    p = Path(file_path).resolve()
    default_graph = dict(nodes=dict(), edges=[])
    if not p.exists():
        return default_graph
    try:
        contents = p.read_text()
        return json.loads(contents)
    except OSError as e:
        console.log(f"⚠️ Error loading graph from {file_path}: {e}", style="warning")
        return default_graph
