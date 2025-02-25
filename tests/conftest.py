# tests/conftest.py

import pytest
import networkx as nx
from unittest.mock import MagicMock
from pathlib import Path
import os

from graph_of_thoughts.context_manager import ContextGraphManager
from graph_of_thoughts.chat_manager import ChatManager
from graph_of_thoughts.models import SeedData


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary output directory for test files."""
    output_dir = Path("tests/temp_output")
    os.makedirs(output_dir, exist_ok=True)
    yield output_dir

    # Cleanup after test
    for file in output_dir.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass

    try:
        output_dir.rmdir()
    except Exception:
        pass


@pytest.fixture(scope="function")
def mock_context_manager():
    """Create a mock context manager for testing."""
    context_manager = MagicMock(spec=ContextGraphManager)

    # Set up common mock behavior
    context_manager.graph_storage = MagicMock()
    context_manager.graph_storage.graph = nx.DiGraph()

    # Mock methods we'll commonly use
    context_manager.query_context.return_value = ["node1", "node2", "node3"]
    context_manager.visualize_graph_as_text.return_value = "Mock graph visualization"

    # Properly set up the tokenizer mock to handle the to() method
    tokenizer_mock = MagicMock()

    # Create a mock dict-like object that also has a to() method
    tokenizer_return = MagicMock()
    tokenizer_return.__getitem__ = lambda self, key: MagicMock()
    tokenizer_return.to = MagicMock(return_value=tokenizer_return)  # Return self from to() method

    tokenizer_mock.return_value = tokenizer_return
    tokenizer_mock.decode.return_value = "Mock LLM response"
    tokenizer_mock.eos_token_id = 0

    # Assign the tokenizer as an attribute
    type(context_manager).tokenizer = tokenizer_mock

    # Mock the model
    model_mock = MagicMock()
    model_mock.generate.return_value = [MagicMock()]  # Mock output tensor
    model_mock.device = "cpu"
    context_manager.model = model_mock

    return context_manager


@pytest.fixture(scope="function")
def sample_graph():
    """Create a sample graph for testing."""
    graph = nx.DiGraph()

    # Add nodes with realistic data structure
    node1_data = {"data": {"content": "Root node content", "importance": 1.0}}
    node2_data = {"data": {"content": "Child node content", "importance": 0.8}}
    node3_data = {"data": {"content": "Leaf node content", "importance": 0.6}}

    graph.add_node("root", **node1_data)
    graph.add_node("child", **node2_data)
    graph.add_node("leaf", **node3_data)

    # Add edges
    graph.add_edge("root", "child")
    graph.add_edge("child", "leaf")

    return graph


@pytest.fixture(scope="function")
def mock_seed_data():
    """Create a mock seed data for testing."""
    return MagicMock(spec=SeedData)


@pytest.fixture(scope="function")
def chat_manager(mock_context_manager):
    """Create a ChatManager instance with a mock context manager."""
    return ChatManager(mock_context_manager)
