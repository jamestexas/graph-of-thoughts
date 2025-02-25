import json
import os
from typing import TYPE_CHECKING
import torch
from transformers import GenerationConfig
from graph_of_thoughts.context_manager import MODEL_NAME, MAX_NEW_TOKENS
from graph_of_thoughts.constants import OUTPUT_DIR, FILES_DIR, console
import logging

from graph_of_thoughts.utils import get_llm_model, get_tokenizer

if TYPE_CHECKING:
    from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_normal_response(
    query: str,
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    generation_config: GenerationConfig,
) -> str:
    """Generates a normal LLM response for a given query."""
    inputs = tokenizer(query, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )
    return result


def process_queries_from_file(
    filepath: str,
    output_filepath: str,
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
) -> None:
    """Reads queries from a file, generates responses, and saves them to an output file incrementally."""

    # Ensure the output directory exists
    if not OUTPUT_DIR.exists():
        console.log(f"Creating missing output dir: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(exist_ok=True)

    if not FILES_DIR.exists():
        console.log(f"Creating missing files dir: {FILES_DIR}")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    try:
        with open(filepath, "r") as f:
            queries = [line.strip() for line in f]
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return

    # Initialize the output file with an empty list
    with open(output_filepath, "w") as f:
        json.dump([], f)

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    for query in queries:
        logging.info(f"Generating response for: {query}")
        response = generate_normal_response(
            query=query,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
        data = dict(
            query=query,
            response=response,
        )

        # Read the existing data, append the new response, and write it back
        try:
            with open(output_filepath, "r+") as f:
                file_data = json.load(f)
                file_data.append(data)
                f.seek(0)
                json.dump(file_data, f, indent=4)
                f.truncate()
        except (FileNotFoundError, json.JSONDecodeError):
            logging.error(f"Error reading/writing to {output_filepath}")
            return

    logging.info(f"Responses saved incrementally to: {output_filepath}")


if __name__ == "__main__":
    model = get_llm_model(model_name=MODEL_NAME)
    tokenizer = get_tokenizer(model_name=MODEL_NAME)
    process_queries_from_file(
        filepath=(FILES_DIR / "input_queries.txt"),
        output_filepath=(OUTPUT_DIR / "normal_outputs.json"),
        model=model,
        tokenizer=tokenizer,
    )
