# graph_of_thoughts/utils.py
import json
import logging
import re
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_of_thoughts.constants import (
    EMBEDDING_MODEL,
    MODEL_NAME,
    SYSTEM_PROMPT,
    console,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


DEVICE = None  # see get_torch_device
_SENTENCE_MODEL_INSTANCE = None  # see get_sentence_transformer
_LLM_MODEL_INSTANCE = None  # see get_llm_model
_TOKENIZER_INSTANCE = None  # see get_tokenizer


def get_torch_device() -> str:
    """Determine and return the available torch device.

    Checks devices in the following order:
        1. MPS
        2. CUDA
        3. CPU

    Returns:
        str: One of "mps", "cuda", or "cpu".
    """
    global DEVICE
    if DEVICE is None:
        console.log("Determining torch device...", style="info")
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        elif torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
        console.log(f"Using torch device: {DEVICE}", style="info")
    return DEVICE


def get_llm_model(model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    """
    Load the LLM model for generating text.
    Uses a singleton pattern to avoid repeatedly loading the model.
    """
    global _LLM_MODEL_INSTANCE
    if _LLM_MODEL_INSTANCE is None:
        console.log(f"Initializing LLM model: {model_name}", style="info")
        _LLM_MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(
            model_name, return_dict_in_generate=True, torch_dtype=torch.float16
        ).to(get_torch_device())
    return _LLM_MODEL_INSTANCE


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """
    Load the tokenizer for the LLM model.
    Uses a singleton pattern to avoid repeatedly loading the tokenizer.
    """
    global _TOKENIZER_INSTANCE
    if _TOKENIZER_INSTANCE is None:
        console.log(f"Initializing tokenizer: {model_name}", style="info")
        _TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(model_name)
        if _TOKENIZER_INSTANCE.pad_token is None:
            _TOKENIZER_INSTANCE.pad_token = _TOKENIZER_INSTANCE.eos_token
    return _TOKENIZER_INSTANCE


def get_sentence_transformer(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load a SentenceTransformer model for embedding text.
    Uses a singleton pattern to avoid repeatedly loading the model.
    """
    global _SENTENCE_MODEL_INSTANCE
    if _SENTENCE_MODEL_INSTANCE is None:
        console.log(
            f"Initializing SentenceTransformer model: {model_name}",
            style="info",
        )
        _SENTENCE_MODEL_INSTANCE = SentenceTransformer(model_name)
    return _SENTENCE_MODEL_INSTANCE


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
    except ValueError as e:
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
                    raise ValueError(f"Extracted JSON is invalid: {e}") from e
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
        raise ValueError(f"Invalid JSON format: {e}") from e


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
