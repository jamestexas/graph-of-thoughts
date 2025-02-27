# graph_of_thoughts/debug_tools.py

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from graph_of_thoughts.constants import console


def inspect_model_output(output: str, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect and analyze the output from an LLM for debugging.

    Args:
        output: The raw output string from the LLM
        save_path: Optional path to save the debug information

    Returns:
        A dictionary containing analysis of the output
    """
    result = {
        "length": len(output),
        "has_internal_tags": "<internal>" in output and "</internal>" in output,
        "has_final_tags": "<final>" in output and "</final>" in output,
        "has_json_tags": "<json>" in output and "</json>" in output,
        "has_code_blocks": "```" in output,
        "first_300_chars": output[:300],
        "last_300_chars": output[-300:] if len(output) > 300 else output,
    }

    # Try to extract JSON
    json_matches = []
    patterns = [
        r"<internal>\s*(\{.*?\})\s*</internal>",
        r"<json>\s*(\{.*?\})\s*</json>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r'(\{[\s\S]*?"nodes"[\s\S]*?"edges"[\s\S]*?\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                json_data = json.loads(json_str)
                json_matches.append({
                    "pattern": pattern,
                    "has_nodes": "nodes" in json_data,
                    "has_edges": "edges" in json_data,
                    "json_length": len(json_str),
                    "json_preview": json_str[:100] + "..." if len(json_str) > 100 else json_str,
                })
            except json.JSONDecodeError as e:
                json_matches.append({
                    "pattern": pattern,
                    "error": str(e),
                    "json_preview": json_str[:100] + "..." if len(json_str) > 100 else json_str,
                })

    result["json_matches"] = json_matches

    # Look for final response sections
    final_sections = []
    final_patterns = [
        r"<final>(.*?)</final>",
        r"Answer:(.*?)(?=$|\n\n)",
        r"Response:(.*?)(?=$|\n\n)",
    ]

    for pattern in final_patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            final_text = match.group(1).strip()
            final_sections.append({
                "pattern": pattern,
                "length": len(final_text),
                "preview": final_text[:100] + "..." if len(final_text) > 100 else final_text,
            })

    result["final_sections"] = final_sections

    # Save to file if requested
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the analysis
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

        # Also save the raw output
        raw_path = f"{save_path}.raw.txt"
        with open(raw_path, "w") as f:
            f.write(output)

        console.log(f"Debug info saved to {save_path}", style="info")
        console.log(f"Raw output saved to {raw_path}", style="info")

    return result


def debug_output_to_console(output: str) -> None:
    """
    Print debug information about model output to the console.

    Args:
        output: The raw output string from the LLM
    """
    analysis = inspect_model_output(output)

    console.log("===== LLM OUTPUT DEBUG INFO =====", style="bold yellow")
    console.log(f"Output length: {analysis['length']} characters", style="info")

    # Tag presence
    tags = []
    if analysis["has_internal_tags"]:
        tags.append("<internal>")
    if analysis["has_final_tags"]:
        tags.append("<final>")
    if analysis["has_json_tags"]:
        tags.append("<json>")
    if analysis["has_code_blocks"]:
        tags.append("```")

    if tags:
        console.log(f"Found tags: {', '.join(tags)}", style="info")
    else:
        console.log("No structural tags found in output", style="warning")

    # JSON content
    if analysis["json_matches"]:
        console.log("JSON content found:", style="info")
        for i, match in enumerate(analysis["json_matches"]):
            if "error" in match:
                console.log(f"  Match {i + 1}: Invalid JSON - {match['error']}", style="error")
            else:
                status = []
                if match["has_nodes"]:
                    status.append("has nodes")
                if match["has_edges"]:
                    status.append("has edges")

                console.log(f"  Match {i + 1}: Valid JSON ({', '.join(status)})", style="info")
                console.log(f"    Preview: {match['json_preview']}", style="dim")
    else:
        console.log("No JSON content found", style="warning")

    # Final sections
    if analysis["final_sections"]:
        console.log("Final response sections found:", style="info")
        for i, section in enumerate(analysis["final_sections"]):
            console.log(f"  Section {i + 1}: {section['length']} chars", style="info")
            console.log(f"    Preview: {section['preview']}", style="dim")
    else:
        console.log("No final response sections found", style="warning")

    # Preview
    console.log("First 300 characters:", style="bold")
    console.log(analysis["first_300_chars"], style="dim")
    console.log("Last 300 characters:", style="bold")
    console.log(analysis["last_300_chars"], style="dim")
    console.log("================================", style="bold yellow")


def add_debug_to_chat_manager(chat_manager, output_dir="debug"):
    """
    Patch a ChatManager instance to add debug functionality.

    Args:
        chat_manager: The ChatManager instance to patch
        output_dir: Directory to save debug outputs
    """
    original_generate = chat_manager.generate_response

    def generate_with_debug(*args, **kwargs):
        response = original_generate(*args, **kwargs)

        # Create debug directory
        debug_dir = Path(output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Generate a timestamp and query hash for the filename
        import hashlib
        import time

        query = args[0] if args else kwargs.get("query", "unknown")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = int(time.time())

        debug_path = debug_dir / f"debug_{timestamp}_{query_hash}.json"

        # Save debug info
        inspect_model_output(response, str(debug_path))

        return response

    # Replace the original method
    chat_manager.generate_response = generate_with_debug

    # Add a debug_output method for easy access
    chat_manager.debug_output = lambda output: debug_output_to_console(output)

    console.log("Debug functionality added to ChatManager", style="info")
    return chat_manager
