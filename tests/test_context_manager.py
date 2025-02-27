from unittest import mock

import networkx as nx
import numpy as np
import pytest

from graph_of_thoughts.context_manager import (
    ChainOfThoughtParser,
    ContextGraphManager,
    FastEmbeddingIndex,
    get_context_mgr,
    parse_chain_of_thought,
    seed_nodes,
    simulate_chat,
)
from graph_of_thoughts.models import ChainOfThought, SeedData


class TestFastEmbeddingIndex:
    def test_initialization(self):
        # Test with default dimension
        index = FastEmbeddingIndex()
        assert index.index.d == 384  # Default dimension
        assert len(index.nodes) == 0

        # Test with custom dimension
        custom_dim = 512
        index = FastEmbeddingIndex(dimension=custom_dim)
        assert index.index.d == custom_dim

    def test_add_node(self):
        index = FastEmbeddingIndex(dimension=2)
        # Add a node
        node_id = "test_node"
        embedding = np.array([1.0, 2.0], dtype=np.float32)
        index.add_node(node_id, embedding)

        # Verify node was added
        assert len(index.nodes) == 1
        if index.nodes[0] != node_id:
            raise AssertionError(f"Expected node_id {node_id}, but got {index.nodes[0]}")
        assert index.index.ntotal == 1

    def test_query(self):
        index = FastEmbeddingIndex(dimension=2)

        # Add multiple nodes
        node_ids = ["node1", "node2", "node3"]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
        ]

        for node_id, embedding in zip(node_ids, embeddings):
            index.add_node(node_id, embedding)

        # Query closest to [0.6, 0.4]
        query_embedding = np.array([0.6, 0.4], dtype=np.float32)
        results = index.query(query_embedding, top_k=2)

        # node3 should be closest, followed by node1
        assert results[0] == "node3"
        assert len(results) == 2


# -----------------------------
# Tests for extract_json_string & balanced extraction
# -----------------------------
def test_extract_json_string_with_json_tag():
    """Test that extraction via a <json> tag works."""
    parser = ChainOfThoughtParser()
    raw = 'Prefix text <json>{"nodes": {"A": "content"}, "edges": []}</json> suffix text'
    result = parser.extract_json_string(raw)
    expected = '{"nodes": {"A": "content"}, "edges": []}'
    assert result == expected


def test_extract_json_string_falls_back_to_balanced():
    """If no regex pattern matches (e.g. missing required keys), use balanced extraction."""
    parser = ChainOfThoughtParser()
    # Input without the words "nodes" and "edges" so that none of the regex patterns match.
    raw = 'Some random text {"foo": "bar"} extra text'
    result = parser.extract_json_string(raw)
    expected = '{"foo": "bar"}'
    assert result == expected


def test_extract_json_balanced_valid():
    """Test the balanced extraction on a string with nested JSON."""
    parser = ChainOfThoughtParser()
    raw = 'Ignore this {"outer": {"inner": "value"}, "list": [1,2,3]} and this'
    result = parser.extract_json_balanced(raw)
    expected = '{"outer": {"inner": "value"}, "list": [1,2,3]}'
    assert result == expected


def test_extract_json_balanced_invalid():
    """Test that an unbalanced JSON input raises ValueError."""
    parser = ChainOfThoughtParser()
    raw = 'Some text {"nodes": {"A": "content" '  # missing closing braces
    with pytest.raises(ValueError, match="No balanced JSON object found."):
        parser.extract_json_balanced(raw)


# -----------------------------
# Tests for load_json
# -----------------------------
def test_load_json_valid():
    """Test loading of a valid JSON string."""
    parser = ChainOfThoughtParser()
    json_str = '{"nodes": {"A": "content"}, "edges": []}'
    result = parser.load_json(json_str)
    assert isinstance(result, dict)
    assert result["nodes"] == {"A": "content"}


def test_load_json_fix():
    """Test that load_json can fix missing quotes on keys."""
    parser = ChainOfThoughtParser()
    # JSON with missing quotes for keys
    json_str = '{nodes: {"A": "content"}, edges: []}'
    result = parser.load_json(json_str)
    assert "nodes" in result
    assert result["nodes"] == {"A": "content"}


def test_load_json_invalid():
    """Test that an unfixable JSON string raises ValueError."""
    parser = ChainOfThoughtParser()
    json_str = '{nodes: {"A": "content", '  # intentionally malformed
    with pytest.raises(ValueError, match="Invalid JSON format."):
        parser.load_json(json_str)


# -----------------------------
# Tests for validate_structure
# -----------------------------
def test_validate_structure_valid():
    """When both 'nodes' and 'edges' exist, validate_structure returns them unchanged."""
    parser = ChainOfThoughtParser()
    data = {"nodes": {"A": "content"}, "edges": []}
    result = parser.validate_structure(data)
    assert result == data


def test_validate_structure_partial():
    """If one of the fields is missing but the other is non-empty, fill in the missing one."""
    parser = ChainOfThoughtParser()
    data = {"nodes": {"A": "content"}}
    result = parser.validate_structure(data)
    # Missing "edges" should default to an empty list.
    assert result == {"nodes": {"A": "content"}, "edges": []}


def test_validate_structure_missing():
    """If neither nodes nor edges are present, raise ValueError."""
    parser = ChainOfThoughtParser()
    data = {"something": "else"}
    with pytest.raises(ValueError, match="JSON missing 'nodes' or 'edges'."):
        parser.validate_structure(data)


# -----------------------------
# Tests for validate_nodes
# -----------------------------
def test_validate_nodes_dict():
    """Test that validate_nodes converts non-string values to strings."""
    parser = ChainOfThoughtParser()
    nodes = {"A": 123, "B": {"key": "value"}}
    result = parser.validate_nodes(nodes)
    assert result["A"] == "123"
    assert result["B"] == "{'key': 'value'}" or result["B"] == '{"key": "value"}'


def test_validate_nodes_list():
    """Test that validate_nodes converts a list of dicts into a proper nodes dict."""
    parser = ChainOfThoughtParser()
    nodes = [
        {"id": "A", "content": "foo"},
        {"name": "B", "description": "bar"},
        {"id": "C"},  # missing content; should fall back to str()
    ]
    result = parser.validate_nodes(nodes)
    assert result["A"] == "foo"
    assert result["B"] == "bar"
    # We expect a fallback key for the third node.
    assert any(k.startswith("node_") for k in result.keys())


# -----------------------------
# Tests for validate_edges
# -----------------------------
def test_validate_edges_list():
    """Test that a valid list of edges is returned unchanged."""
    parser = ChainOfThoughtParser()
    edges = [["A", "B"], ["B", "C"]]
    result = parser.validate_edges(edges)
    assert result == [["A", "B"], ["B", "C"]]


def test_validate_edges_dict():
    """Test conversion of a dict-form edges to a list."""
    parser = ChainOfThoughtParser()
    edges = {"A": ["B", "C"], "X": "Y"}
    result = parser.validate_edges(edges)
    expected = [["A", "B"], ["A", "C"], ["X", "Y"]]
    # Order might vary, so compare sorted lists.
    assert sorted(result) == sorted(expected)


def test_validate_edges_invalid():
    """Test that invalid edge formats are skipped."""
    parser = ChainOfThoughtParser()
    edges = [
        ["A", "B"],
        {"source": "B", "target": "C"},
        "invalid edge",
        [1, 2, 3],
        {"not": "an edge"},
    ]
    result = parser.validate_edges(edges)
    # Only the first two valid edges should be kept.
    assert result == [["A", "B"], ["B", "C"]]


# -----------------------------
# Integration tests
# -----------------------------
def test_parse_integration():
    """Test that the full parse() method extracts, loads, validates, and returns a ChainOfThought."""
    parser = ChainOfThoughtParser()
    raw_output = """
        Preamble text.
        <json>{
            "nodes": {"A": "content A", "B": "content B"},
            "edges": [{"source": "A", "target": "B"}]
        }</json>
        Postamble text.
    """
    result = parser.parse(raw_output)
    assert isinstance(result, ChainOfThought)
    # Expect that nodes were not altered (they are already strings)
    assert result.nodes == {"A": "content A", "B": "content B"}
    # The dict-form edge should have been converted to a list.
    assert result.edges == [["A", "B"]]


def test_parse_chain_of_thought_function():
    """Test the public function wrapper for parsing."""
    raw_output = '{"nodes": {"A": "content A"}, "edges": [["A", "B"]]}'
    result = parse_chain_of_thought(raw_output)
    assert isinstance(result, ChainOfThought)
    assert result.nodes == {"A": "content A"}
    assert result.edges == [["A", "B"]]


# TODO: Make these mocks more consistent / less repetitive
class TestContextGraphManager:
    def test_initialization_with_defaults(self, mock_context_manager):
        # Create a basic DiGraph
        initial_graph = nx.DiGraph()
        initial_graph.add_node("A", data={"content": "A content"})
        initial_graph.add_node("B", data={"content": "B content"})
        initial_graph.add_edge("A", "B")

        # Test with our mocked context
        with (
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph") as mock_build,
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine") as mock_embedding,
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine") as mock_reasoning,
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer") as mock_tokenizer,
            mock.patch("graph_of_thoughts.context_manager.get_llm_model") as mock_model,
            mock.patch(
                "graph_of_thoughts.context_manager.get_sentence_transformer"
            ) as mock_get_sentence,
        ):
            # Set up mocks
            mock_build.return_value = initial_graph

            # Mock storage and graph
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance
            mock_storage_instance.graph = initial_graph

            # Mock embedding engine
            mock_embedding_instance = mock.MagicMock()
            mock_embedding.return_value = mock_embedding_instance

            # Mock reasoning engine
            mock_reasoning_instance = mock.MagicMock()
            mock_reasoning.return_value = mock_reasoning_instance

            # Create manager
            manager = ContextGraphManager()

            # Check initialization
            mock_build.assert_called_once()
            mock_storage.assert_called_once()
            mock_embedding.assert_called_once()
            mock_reasoning.assert_called_once()
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()
            mock_get_sentence.assert_called_once()

            # Check graph assignment
            assert manager.graph is mock_storage_instance.graph

    def test_add_context(self):
        # Test add_context method with mocks
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine") as mock_embedding,
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance

            mock_embedding_instance = mock.MagicMock()
            mock_embedding.return_value = mock_embedding_instance

            # Create manager
            manager = ContextGraphManager()

            # Call add_context
            node_id = "test_node"
            content = "Test content"
            metadata = {"key": "value"}

            manager.add_context(node_id, content, metadata)

            # Verify calls
            mock_storage_instance.add_node.assert_called_once_with(node_id, content, metadata)
            mock_embedding_instance.add_node_embedding.assert_called_once_with(node_id, content)

    def test_query_context(self):
        # Test query_context method with mocks
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage"),
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine") as mock_embedding,
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_embedding_instance = mock.MagicMock()
            mock_embedding.return_value = mock_embedding_instance

            # Set up expected result
            expected_nodes = ["node1", "node2", "node3"]
            mock_embedding_instance.find_similar_nodes.return_value = expected_nodes

            # Create manager
            manager = ContextGraphManager()

            # Call query_context
            query = "Test query"
            top_k = 5
            result = manager.query_context(query, top_k)

            # Verify calls and result
            mock_embedding_instance.find_similar_nodes.assert_called_once_with(query, top_k)
            assert result == expected_nodes

    def test_visualize_graph_as_text(self):
        """Test visualize_graph_as_text method with mocks."""
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance

            # Set up expected result
            mock_storage_instance.visualize_as_text.return_value = "Mocked graph visualization"

            # Create manager
            manager = ContextGraphManager()

            # Call visualize_graph_as_text
            result = manager.visualize_graph_as_text()

            # This might need additional mocking for the nodes and edges
            assert isinstance(result, str)
            assert "**Current Knowledge Graph**" in result

    def test_graph_to_json(self):
        # Test graph_to_json method with mocks
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance

            # Set up expected result - as a dictionary, not a string
            expected_result = {"nodes": {}, "edges": []}
            mock_storage_instance.to_json.return_value = expected_result

            # Create manager
            manager = ContextGraphManager()

            # Call graph_to_json
            result = manager.graph_to_json()

            # Verify calls and result
            mock_storage_instance.to_json.assert_called_once()
            assert result == expected_result

    def test_decay_importance(self):
        # Test decay_importance method with mocks
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance

            # Create a simple graph with nodes
            mock_graph = nx.DiGraph()
            mock_graph.add_nodes_from(["node1", "node2", "node3"])
            mock_storage_instance.graph = mock_graph

            # Create manager
            manager = ContextGraphManager()

            # Call decay_importance
            decay_factor = 0.9
            manager.decay_importance(decay_factor=decay_factor)

            # Verify decay was called for each node
            assert mock_storage_instance.decay_node_importance.call_count == 3
            mock_storage_instance.decay_node_importance.assert_any_call(
                node_id="node1", decay_factor=decay_factor
            )
            mock_storage_instance.decay_node_importance.assert_any_call(
                node_id="node2", decay_factor=decay_factor
            )
            mock_storage_instance.decay_node_importance.assert_any_call(
                node_id="node3", decay_factor=decay_factor
            )

    def test_prune_context(self):
        # Test prune_context method with mocks
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage") as mock_storage,
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine"),
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_storage_instance = mock.MagicMock()
            mock_storage.return_value = mock_storage_instance

            # Create manager
            manager = ContextGraphManager()

            # Call prune_context
            threshold = 0.3
            manager.prune_context(threshold=threshold)

            # Verify prune was called
            mock_storage_instance.prune_low_importance_nodes.assert_called_once_with(threshold)

    def test_iterative_refinement_success(self):
        """Test iterative_refinement method with successful parsing."""
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage"),
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine") as mock_reasoning,
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_reasoning_instance = mock.MagicMock()
            mock_reasoning.return_value = mock_reasoning_instance

            # Create manager
            manager = ContextGraphManager()

            # Call iterative_refinement with JSON string input
            reasoning_output = '<json>{"nodes": {"A": "Node A content", "B": "Node B content"}, "edges": [["A", "B"]]}</json>'

            with mock.patch(
                "graph_of_thoughts.context_manager.parse_chain_of_thought"
            ) as mock_parse:
                chain_obj = ChainOfThought(
                    nodes={"A": "Node A content", "B": "Node B content"},
                    edges=[["A", "B"]],
                )
                mock_parse.return_value = chain_obj

                # Test the method
                manager.iterative_refinement(reasoning_output)

                # Verify that parse_chain_of_thought was called
                mock_parse.assert_called_once()

                # Verify that the reasoning engine was updated
                mock_reasoning_instance.update_from_chain_of_thought.assert_called_once()

    def test_iterative_refinement_failure(self):
        # Test iterative_refinement method with parsing failure
        with (
            mock.patch("graph_of_thoughts.context_manager.GraphStorage"),
            mock.patch("graph_of_thoughts.context_manager.EmbeddingEngine"),
            mock.patch("graph_of_thoughts.context_manager.ReasoningEngine") as mock_reasoning,
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer"),
            mock.patch("graph_of_thoughts.context_manager.get_llm_model"),
            mock.patch("graph_of_thoughts.context_manager.get_sentence_transformer"),
            mock.patch("graph_of_thoughts.context_manager.build_initial_graph"),
        ):
            # Create mock instances
            mock_reasoning_instance = mock.MagicMock()
            mock_reasoning.return_value = mock_reasoning_instance

            # Create manager
            manager = ContextGraphManager()

            # Call iterative_refinement with invalid output
            reasoning_output = "Invalid output"

            with (
                mock.patch(
                    "graph_of_thoughts.context_manager.parse_chain_of_thought"
                ) as mock_parse,
                mock.patch("graph_of_thoughts.context_manager.console.log") as mock_log,
            ):
                mock_parse.side_effect = ValueError("Test error")

                manager.iterative_refinement(reasoning_output)

                # Verify error was logged
                mock_log.assert_called_once()
                mock_reasoning_instance.update_from_chain_of_thought.assert_not_called()


class TestHelperFunctions:
    def test_get_context_mgr(self):
        """Test get_context_mgr function with proper mocking."""
        with (
            mock.patch("graph_of_thoughts.context_manager.get_unified_llm_model") as mock_get_model,
            mock.patch(
                "graph_of_thoughts.context_manager.ContextGraphManager"
            ) as mock_manager_class,
        ):
            # Set up mock model
            mock_model_instance = mock.MagicMock()
            # Mock the tokenizer attribute directly on the model
            mock_model_instance.tokenizer = mock.MagicMock()
            mock_get_model.return_value = mock_model_instance

            # Call function with dummy model name
            get_context_mgr("test_model")

            # Verify calls
            mock_get_model.assert_called_once_with(backend="hf", model_name="test_model")
            mock_manager_class.assert_called_once()

            # Check that the call to ContextGraphManager includes the right parameters
            _, kwargs = mock_manager_class.call_args
            assert "tokenizer" in kwargs
            assert "model" in kwargs
            assert kwargs["model"] is mock_model_instance
            assert kwargs["tokenizer"] is mock_model_instance.tokenizer

    def test_seed_nodes(self):
        # Test seed_nodes function
        mock_manager = mock.MagicMock()

        # Create seed data
        seed_data = [
            SeedData(node_id="node1", content="Content 1", metadata={"key1": "value1"}),
            SeedData(node_id="node2", content="Content 2", metadata={"key2": "value2"}),
        ]

        # Call function
        seed_nodes(mock_manager, seed_data)

        # Verify calls
        assert mock_manager.add_context.call_count == 2
        mock_manager.add_context.assert_any_call("node1", "Content 1", {"key1": "value1"})
        mock_manager.add_context.assert_any_call("node2", "Content 2", {"key2": "value2"})

    def test_simulate_chat(self):
        # Test simulate_chat function
        mock_manager = mock.MagicMock()

        # Import instead of patching directly
        with (
            mock.patch("graph_of_thoughts.chat_manager.ChatManager") as mock_chat_manager_class,
            mock.patch("graph_of_thoughts.context_manager.seed_nodes") as mock_seed_nodes,
            mock.patch("graph_of_thoughts.context_manager.console.log") as mock_log,
        ):
            # Set up mock ChatManager
            mock_chat_manager = mock.MagicMock()
            mock_chat_manager_class.return_value = mock_chat_manager
            mock_chat_manager.simulate_conversation.return_value = [
                "result1",
                "result2",
            ]

            # Create inputs
            conversation_inputs = ["input1", "input2"]
            seed_data = [
                SeedData(node_id="node1", content="Content 1", metadata={"importance": 1.0})
            ]

            # Call function
            simulate_chat(mock_manager, conversation_inputs, seed_data, "test_experiment")

            # Verify calls
            mock_chat_manager_class.assert_called_once()
            mock_seed_nodes.assert_called_once_with(mock_manager, seed_data)
            mock_chat_manager.simulate_conversation.assert_called_once_with(
                inputs=conversation_inputs, experiment_name="test_experiment"
            )
            mock_log.asseert_called()

    def test_simulate_chat_no_seed(self):
        # Test simulate_chat function without seed data
        mock_manager = mock.MagicMock()

        # Import instead of patching directly
        with (
            mock.patch("graph_of_thoughts.chat_manager.ChatManager") as mock_chat_manager_class,
            mock.patch("graph_of_thoughts.context_manager.seed_nodes") as mock_seed_nodes,
            mock.patch("graph_of_thoughts.context_manager.console.log"),
        ):
            # Set up mock ChatManager
            mock_chat_manager = mock.MagicMock()
            mock_chat_manager_class.return_value = mock_chat_manager

            # Create inputs (no seed data)
            conversation_inputs = ["input1", "input2"]

            # Call function - with module-level import mocked
            with mock.patch.object(mock_manager, "__class__", create=True):
                simulate_chat(mock_manager, conversation_inputs)

            # Verify calls
            mock_chat_manager_class.assert_called_once()
            mock_seed_nodes.assert_not_called()  # Should not be called without seed data
            mock_chat_manager.simulate_conversation.assert_called_once()
