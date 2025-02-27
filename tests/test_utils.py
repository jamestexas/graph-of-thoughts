from unittest.mock import MagicMock, patch

import pytest

from graph_of_thoughts import utils
from graph_of_thoughts.utils import (
    build_llama_instruct_prompt,
    build_structured_prompt,
    clean_text,
    extract_and_clean_json,
    extract_balanced_json,
    extract_json_substring,
    get_llm_model,
    get_sentence_transformer,
    get_tokenizer,
    get_torch_device,
    summarize_text,
    trim_prompt,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("  This   is  a    test.  ", "This is a test."),
        ("\nNew\n\nline\n\tcharacters\n", "New line characters"),
        (
            "Multiple   spaces    should   be  removed",
            "Multiple spaces should be removed",
        ),
    ],
)
def test_clean_text(input_text, expected_output):
    assert clean_text(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, max_sentences, expected_output",
    [
        (
            "This is a test. Another sentence here. And another one.",
            1,
            "This is a test.",
        ),
        (
            "This is a test! Another one? And another.",
            2,
            "This is a test! Another one?",
        ),
        ("Only one sentence.", 3, "Only one sentence."),
    ],
)
def test_summarize_text(input_text, max_sentences, expected_output):
    assert summarize_text(input_text, max_sentences) == expected_output


@pytest.mark.parametrize(
    "raw_output, expected_json",
    [
        ('<json>{"nodes": [], "edges": []}</json>', {"nodes": [], "edges": []}),
        (
            'Some text before <json>{"nodes": [1], "edges": [2]}</json> more text',
            {"nodes": [1], "edges": [2]},
        ),
        ('{"nodes": ["A"], "edges": ["B"]}', {"nodes": ["A"], "edges": ["B"]}),
    ],
)
def test_extract_json_substring(raw_output, expected_json):
    assert extract_json_substring(raw_output) == expected_json


def test_extract_json_substring_invalid():
    assert extract_json_substring("No JSON here") is None


@pytest.mark.parametrize(
    "text, expected_json",
    [
        ('{"key": "value"}', '{"key": "value"}'),
        ('Some text {"key": "value"} more text', '{"key": "value"}'),
        ('{"nodes": [1], "edges": [2]}', '{"nodes": [1], "edges": [2]}'),
    ],
)
def test_extract_balanced_json(text, expected_json):
    assert extract_balanced_json(text) == expected_json


def test_extract_balanced_json_unbalanced():
    with pytest.raises(ValueError):
        extract_balanced_json("{ unbalanced json")


@pytest.mark.parametrize(
    "text, expected_json",
    [
        ('<json>{"test": "data"}</json>', '{\n  "test": "data"\n}'),
        ('```json\n{"test": "data"}\n```', '{\n  "test": "data"\n}'),
    ],
)
def test_extract_and_clean_json(text, expected_json):
    assert extract_and_clean_json(text) == expected_json


def test_extract_and_clean_json_invalid():
    with pytest.raises(ValueError):
        extract_and_clean_json("No JSON here")


@pytest.mark.parametrize(
    "system_text, user_text, expected_output",
    [
        (
            "System message",
            "User input",
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nSystem message\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser input\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        ),
    ],
)
def test_build_llama_instruct_prompt(system_text, user_text, expected_output):
    assert build_llama_instruct_prompt(system_text, user_text) == expected_output


@pytest.mark.parametrize(
    "query",
    [
        "What is the meaning of life?",
        "Explain caching strategies.",
    ],
)
def test_build_structured_prompt(query):
    prompt = build_structured_prompt(query)
    expected_strings = [
        query,
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
    ]
    for expected_string in expected_strings:
        assert expected_string in prompt


@pytest.fixture(autouse=True)
def reset_device_singleton():
    """Reset the DEVICE singleton before each test."""
    # Store original
    original_device = utils.DEVICE

    # Reset for the test
    utils.DEVICE = None

    yield

    # Restore after test
    utils.DEVICE = original_device


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: list(range(len(x)))
    return tokenizer


def test_trim_prompt(mock_tokenizer):
    long_prompt = "This is a very long prompt that should be trimmed."
    trimmed = trim_prompt(long_prompt, mock_tokenizer, max_tokens=10)
    assert len(mock_tokenizer.encode(trimmed)) <= 10


@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.cuda.is_available", return_value=False)
def test_get_torch_device_mps(mock_mps, mock_cuda):
    assert get_torch_device() == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=True)
def test_get_torch_device_cuda(mock_mps, mock_cuda):
    assert get_torch_device() == "cuda"


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
def test_get_torch_device_cpu(mock_mps, mock_cuda):
    assert get_torch_device() == "cpu"


@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_get_llm_model(mock_model):
    mock_model.return_value = MagicMock()
    assert get_llm_model()


@patch("transformers.AutoTokenizer.from_pretrained")
def test_get_tokenizer(mock_tokenizer):
    mock_tokenizer.return_value = MagicMock()
    tokenizer = get_tokenizer()
    assert tokenizer is not None


@patch("sentence_transformers.SentenceTransformer")
def test_get_sentence_transformer(mock_transformer):
    mock_transformer.return_value = MagicMock()
    assert get_sentence_transformer()
