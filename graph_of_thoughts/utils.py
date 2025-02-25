# graph_of_thoughts/utils.py
import torch
import re
import json
import logging
from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from graph_of_thoughts.constants import (
    EMBEDDING_MODEL,
    SYSTEM_PROMPT,
    MODEL_NAME,
    console,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_torch_device():
    # Default MPS -> CUDA -> CPU

    if torch.backends.mps.is_available():
        console.log("MPS is available. Using MPS.")
        DEVICE = "mps"
    elif torch.cuda.is_available():
        console.log("CUDA is available. Using GPU.")
        DEVICE = "cuda"
    else:
        console.log("CUDA is not available. Using CPU.")
        DEVICE = "cpu"
    return DEVICE


DEVICE = get_torch_device()


def get_llm_model(model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    """Load the LLM model for generating text."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, return_dict_in_generate=True, torch_dtype=torch.float16
    ).to(DEVICE)


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load the tokenizer for the LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_sentence_transformer(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a SentenceTransformer model for embedding text."""
    return SentenceTransformer(model_name)


# Text processing utilities
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines."""
    return re.sub(r"\s+", " ", text).strip()


def summarize_text(text: str, max_sentences: int = 1) -> str:
    """Extract first N sentences from text as a summary."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences])


# JSON extraction utilities
def extract_json_substring(raw_output: str) -> dict[str, Any] | None:
    """
    Extracts and validates JSON from an LLM response while logging issues.
    """
    logger.debug(f"🔍 Raw LLM Output:\n{raw_output}")

    # Normalize inconsistent JSON markers
    raw_output = raw_output.replace("$json</json>", "").replace("$json", "").strip()

    # First try: Extract properly wrapped JSON
    json_tags_regex = re.compile(r"<json>\s*(\{.*?\})\s*</json>", re.DOTALL)
    if not (match := json_tags_regex.search(raw_output)):
        logger.warning("⚠️ No <json>...</json> tags found. Trying direct extraction...")
        match = re.search(r"(\{.*?\})", raw_output, re.DOTALL)

    if not match:
        logger.error("❌ No valid JSON found in output.")
        return None

    json_str = match.group(1).strip()
    logger.debug(f"✅ Extracted JSON Candidate:\n{json_str}")

    # Ensure JSON contains required fields
    try:
        parsed_json = json.loads(json_str)

        # Check if required fields exist
        if "nodes" not in parsed_json or "edges" not in parsed_json:
            logger.error(f"❌ Missing required fields in JSON: {parsed_json}")
            return None

        logger.info("✅ Successfully parsed and validated JSON!")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing failed: {e}")
        return None


def extract_balanced_json(text: str) -> str:
    """
    Extracts the first balanced JSON object from the given text.

    Args:
        text (str): The text containing a JSON block.

    Returns:
        str: The extracted JSON string.

    Raises:
        ValueError: If no balanced JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the text.")
    stack = []
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            stack.append("{")
        elif char == "}":
            stack.pop()
            if not stack:
                candidate = text[start : i + 1]
                try:
                    # Re-serialize to ensure it's clean
                    parsed = json.loads(candidate)
                    return json.dumps(parsed)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Extracted JSON is invalid: {e}")
    raise ValueError("Unbalanced JSON braces in the text.")


def extract_and_clean_json(text: str) -> str:
    """Extract JSON from LLM response, handling various formats."""
    # Try to extract JSON inside <json>...</json> tags
    json_pattern = re.compile(r"<json>(.*?)</json>", re.DOTALL)
    match = json_pattern.search(text)

    # If not found, try markdown code blocks
    if not match:
        json_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        match = json_pattern.search(text)

    if not match:
        raise ValueError("No JSON found in response")

    json_str = match.group(1).strip()

    # Further parse to find the first valid JSON object
    try:
        # Find first { and matching }
        start = json_str.find("{")
        if start == -1:
            raise ValueError("No JSON object found in extracted text")

        # Track braces for proper JSON detection
        brace_count = 0
        for i, char in enumerate(json_str[start:], start=start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_obj = json_str[start : i + 1]
                    # Validate and pretty-print
                    parsed = json.loads(json_obj)
                    return json.dumps(parsed, indent=2)

        raise ValueError("Unbalanced JSON braces")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


# Prompt building utilities
def build_llama_instruct_prompt(system_text: str, user_text: str) -> str:
    """
    Constructs a multi-turn style prompt for Llama 3.x Instruct models.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_text}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_structured_prompt(query: str) -> str:
    """Build a prompt for structured reasoning about a query."""

    return f"""{SYSTEM_PROMPT}

📌 IMPORTANT: **DO NOT** provide explanations. Output **ONLY** JSON. 

Question: {query}
<json>
"""


def trim_prompt(extended_prompt: str, tokenizer, max_tokens: int = 800) -> str:
    """
    Ensure the extended prompt does not exceed a maximum token count.
    """
    tokens = tokenizer.encode(extended_prompt)
    if len(tokens) <= max_tokens:
        return extended_prompt
    lines = extended_prompt.split("\n")
    trimmed_lines = []
    current_tokens = 0
    for line in lines:
        line_tokens = tokenizer.encode(line)
        if current_tokens + len(line_tokens) <= max_tokens:
            trimmed_lines.append(line)
            current_tokens += len(line_tokens)
        else:
            break
    return "\n".join(trimmed_lines)
