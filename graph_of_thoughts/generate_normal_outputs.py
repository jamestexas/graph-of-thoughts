import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from context_manager import MODEL_NAME
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load your LLM (replace with your model path)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
    "mps"
)  # Or whatever device you are using.


def generate_normal_response(query: str) -> str:
    """Generates a normal LLM response for a given query."""
    inputs = tokenizer(query, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def process_queries_from_file(filepath: str, output_filepath: str):
    """Reads queries from a file, generates responses, and saves them to an output file incrementally."""

    # Ensure the output directory exists
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

    for query in queries:
        logging.info(f"Generating response for: {query}")
        response = generate_normal_response(query)
        data = {"query": query, "response": response}

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
    process_queries_from_file("files/input_queries.txt", "output/normal_outputs.json")
