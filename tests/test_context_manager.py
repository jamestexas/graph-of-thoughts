from unittest import mock

import networkx as nx
import numpy as np
import pytest

from graph_of_thoughts.context_manager import (
    ContextGraphManager,
    FastEmbeddingIndex,
    chat_entry,
    generate_with_context,
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
        assert index.nodes[0] == node_id
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


class TestParseChainOfThought:
    def test_valid_json(self):
        # Test with valid JSON input - using correct structure
        valid_input = """Some text before
        <json>{"nodes": {"A": "Node A content", "B": "Node B content"}, "edges": [["A", "B"]]}</json>
        Some text after"""

        with mock.patch("graph_of_thoughts.context_manager.console.log"):
            # Need to patch the regex search to test our function
            with mock.patch("re.compile") as mock_compile:
                mock_pattern = mock.MagicMock()
                mock_compile.return_value = mock_pattern
                mock_match = mock.MagicMock()
                mock_pattern.search.return_value = mock_match
                mock_match.group.return_value = '{"nodes": {"A": "Node A content", "B": "Node B content"}, "edges": [["A", "B"]]}'

                result = parse_chain_of_thought(valid_input)

        assert isinstance(result, ChainOfThought)
        assert "A" in result.nodes
        assert "B" in result.nodes
        assert result.nodes["A"] == "Node A content"
        assert result.edges[0] == ["A", "B"]

    def test_no_json_tags(self):
        # Test with missing JSON tags
        invalid_input = (
            """{"nodes": {"A": {}, "B": {}}, "edges": [{"source": "A", "target": "B"}]}"""
        )

        with mock.patch("graph_of_thoughts.context_manager.console.log"):
            # Simulate no match found
            with mock.patch("re.compile") as mock_compile:
                mock_pattern = mock.MagicMock()
                mock_compile.return_value = mock_pattern
                mock_pattern.search.return_value = None

                with pytest.raises(ValueError, match="No valid JSON block found"):
                    parse_chain_of_thought(invalid_input)

    def test_invalid_json_syntax(self):
        # Test with invalid JSON syntax
        invalid_input = (
            """<json>{"nodes": {"A": {}, "B": {}}, "edges": [{"source": "A", "target": "B"</json>"""
        )

        with mock.patch("graph_of_thoughts.context_manager.console.log"):
            # Simulate the regex finding a match but JSON parsing failing
            with mock.patch("re.compile") as mock_compile:
                mock_pattern = mock.MagicMock()
                mock_compile.return_value = mock_pattern
                mock_match = mock.MagicMock()
                mock_pattern.search.return_value = mock_match
                mock_match.group.return_value = (
                    '{"nodes": {"A": {}, "B": {}}, "edges": [{"source": "A", "target": "B"'
                )

                with pytest.raises(ValueError, match="Invalid JSON format"):
                    parse_chain_of_thought(invalid_input)

    def test_missing_required_fields(self):
        # Test with JSON missing required fields
        invalid_input = """<json>{"something": "else"}</json>"""

        with mock.patch("graph_of_thoughts.context_manager.console.log"):
            # Simulate the regex and JSON parsing correctly, but missing fields
            with mock.patch("re.compile") as mock_compile:
                mock_pattern = mock.MagicMock()
                mock_compile.return_value = mock_pattern
                mock_match = mock.MagicMock()
                mock_pattern.search.return_value = mock_match
                mock_match.group.return_value = '{"something": "else"}'

                with pytest.raises(ValueError, match="JSON missing 'nodes' or 'edges'"):
                    parse_chain_of_thought(invalid_input)


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
        # Test visualize_graph_as_text method with mocks
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
            expected_text = "Graph visualization"
            mock_storage_instance.visualize_as_text.return_value = expected_text

            # Create manager
            manager = ContextGraphManager()

            # Call visualize_graph_as_text
            result = manager.visualize_graph_as_text()

            # Verify calls and result
            mock_storage_instance.visualize_as_text.assert_called_once()
            assert result == expected_text

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
        # Test iterative_refinement method with successful parsing
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

            # Call iterative_refinement
            reasoning_output = '<json>{"nodes": {"A": "Node A content", "B": "Node B content"}, "edges": [["A", "B"]]}</json>'

            with mock.patch(
                "graph_of_thoughts.context_manager.parse_chain_of_thought"
            ) as mock_parse:
                chain_obj = ChainOfThought(
                    nodes={"A": "Node A content", "B": "Node B content"},
                    edges=[["A", "B"]],
                )
                mock_parse.return_value = chain_obj

                manager.iterative_refinement(reasoning_output)

                # Verify calls
                mock_parse.assert_called_once_with(reasoning_output)
                mock_reasoning_instance.update_from_chain_of_thought.assert_called_once_with(
                    chain_obj
                )

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
    def test_generate_with_context(self):
        # Test generate_with_context function
        mock_manager = mock.MagicMock()
        mock_manager.query_context.return_value = ["node1", "node2", "node3"]
        mock_manager.visualize_graph_as_text.return_value = "Graph visualization"

        # Set up the tokenizer mock correctly
        mock_tokenizer_result = mock.MagicMock()
        mock_tokenizer_result.to.return_value = mock_tokenizer_result
        mock_manager.tokenizer.return_value = mock_tokenizer_result

        # Set up pad_token_id to be a proper integer
        mock_manager.tokenizer.eos_token_id = 1  # Integer not MagicMock

        # Mock the generate method correctly
        mock_output = mock.MagicMock()
        mock_manager.model.generate.return_value = [mock_output]
        mock_manager.tokenizer.decode.return_value = "Generated response"

        # Patch GenerationConfig to avoid validation issues
        with (
            mock.patch("graph_of_thoughts.context_manager.console.log"),
            mock.patch("graph_of_thoughts.context_manager.GenerationConfig") as mock_gen_config,
        ):
            # Set up the mock config
            mock_config = mock.MagicMock()
            mock_gen_config.return_value = mock_config

            result = generate_with_context("Test query", mock_manager)

        # Check the result
        assert result == "Generated response"
        mock_manager.query_context.assert_called_once()
        mock_manager.visualize_graph_as_text.assert_called_once()
        mock_manager.tokenizer.assert_called_once()
        mock_manager.model.generate.assert_called_once()
        mock_manager.tokenizer.decode.assert_called_once()

    def test_get_context_mgr(self):
        # Test get_context_mgr function
        with (
            mock.patch("graph_of_thoughts.context_manager.get_llm_model") as mock_get_model,
            mock.patch("graph_of_thoughts.context_manager.get_tokenizer") as mock_get_tokenizer,
            mock.patch(
                "graph_of_thoughts.context_manager.ContextGraphManager"
            ) as mock_manager_class,
        ):
            # Set up mocks
            mock_model = mock.MagicMock()
            mock_tokenizer = mock.MagicMock()
            mock_get_model.return_value = mock_model
            mock_get_tokenizer.return_value = mock_tokenizer

            # Call function
            get_context_mgr("test_model")

            # Verify calls
            mock_get_model.assert_called_once_with(model_name="test_model")
            mock_get_tokenizer.assert_called_once_with(model_name="test_model")
            mock_manager_class.assert_called_once_with(tokenizer=mock_tokenizer, model=mock_model)

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

    def test_chat_entry(self):
        # Test chat_entry function
        mock_manager = mock.MagicMock()

        with (
            mock.patch("graph_of_thoughts.context_manager.console.log"),
            mock.patch("graph_of_thoughts.context_manager.generate_with_context") as mock_generate,
            mock.patch("graph_of_thoughts.context_manager.extract_and_clean_json") as mock_extract,
        ):
            # Set up mocks
            mock_generate.return_value = "LLM response"
            mock_extract.return_value = "JSON response"

            # Call function
            chat_entry(mock_manager, "User input", 1)

            # Verify calls
            mock_manager.decay_importance.assert_called_once()
            mock_manager.add_context.assert_any_call("user_1", "User input")
            mock_manager.query_context.assert_called_once_with("User input", top_k=3)
            mock_generate.assert_called_once_with("User input", context_manager=mock_manager)
            mock_manager.add_context.assert_any_call("llm_1", "LLM response")
            mock_extract.assert_called_once_with("LLM response")
            mock_manager.iterative_refinement.assert_called_once_with("JSON response")
            mock_manager.prune_context.assert_called_once()

    def test_chat_entry_error_handling(self):
        # Test chat_entry with error in JSON extraction
        mock_manager = mock.MagicMock()

        with (
            mock.patch("graph_of_thoughts.context_manager.console.log") as mock_log,
            mock.patch("graph_of_thoughts.context_manager.generate_with_context") as mock_generate,
            mock.patch("graph_of_thoughts.context_manager.extract_and_clean_json") as mock_extract,
        ):
            # Set up mocks with error
            mock_generate.return_value = "LLM response"
            mock_extract.side_effect = ValueError("Invalid JSON")

            # Call function
            chat_entry(mock_manager, "User input", 1)

            # Verify error handling
            mock_manager.add_context.assert_any_call("llm_1", "LLM response")
            mock_extract.assert_called_once_with("LLM response")
            mock_manager.iterative_refinement.assert_not_called()  # Should not be called due to error
            mock_log.assert_any_call(mock.ANY, style="warning")  # Warning should be logged
            mock_manager.prune_context.assert_called_once()  # Should still be called despite error

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
