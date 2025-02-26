"""Tests for the graph_utils module."""

import json
import os
import tempfile
from unittest import mock

from graph_of_thoughts.graph_utils import (
    add_chain_of_thought_to_graph,
    load_graph,
    update_and_save_graph,
)
from graph_of_thoughts.models import ChainOfThought


class TestAddChainOfThoughtToGraph:
    def test_add_chain_of_thought_to_graph(self):
        """Test adding a chain of thought to the graph."""
        # Create a mock context manager
        mock_context_manager = mock.MagicMock()

        # Create a chain of thought
        chain_obj = ChainOfThought(nodes={"A": "Node A", "B": "Node B"}, edges=[["A", "B"]])

        # Call the function
        with mock.patch("graph_of_thoughts.graph_utils.console.log"):
            add_chain_of_thought_to_graph(chain_obj, mock_context_manager)

        # Check that context manager methods were called
        assert mock_context_manager.add_context.call_count == 2
        mock_context_manager.add_context.assert_any_call(
            "reason_A", "Node A", metadata={"importance": 1.0}
        )
        mock_context_manager.add_context.assert_any_call(
            "reason_B", "Node B", metadata={"importance": 1.0}
        )

        # Check that edge was added
        mock_context_manager.graph_storage.add_edge.assert_called_once_with("reason_A", "reason_B")


class TestUpdateAndSaveGraph:
    def test_empty_input(self):
        """Test with empty input."""
        with mock.patch("graph_of_thoughts.graph_utils.console.log"):
            result = update_and_save_graph(None, "output.json", "")
            assert result is False

    def test_invalid_json(self):
        """Test with invalid JSON input."""
        with (
            mock.patch("graph_of_thoughts.graph_utils.console.log"),
            mock.patch("graph_of_thoughts.graph_utils.parse_chain_of_thought") as mock_parse,
        ):
            mock_parse.side_effect = ValueError("Invalid JSON")

            result = update_and_save_graph(None, "output.json", "invalid")
            assert result is False

    def test_missing_nodes_or_edges(self):
        """Test with missing nodes or edges."""
        with (
            mock.patch("graph_of_thoughts.graph_utils.console.log"),
            mock.patch("graph_of_thoughts.graph_utils.parse_chain_of_thought") as mock_parse,
        ):
            # Create a chain with empty nodes
            mock_parse.return_value = ChainOfThought(nodes={}, edges=[["A", "B"]])

            result = update_and_save_graph(None, "output.json", "valid")
            assert result is False

    def test_new_file_creation(self):
        """Test creating a new file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.json")

            with (
                mock.patch("graph_of_thoughts.graph_utils.console.log"),
                mock.patch("graph_of_thoughts.graph_utils.parse_chain_of_thought") as mock_parse,
            ):
                # Create a valid chain
                mock_parse.return_value = ChainOfThought(
                    nodes={"A": "Node A", "B": "Node B"}, edges=[["A", "B"]]
                )

                # Call the function
                result = update_and_save_graph(None, output_path, "valid")

                # Check result
                assert result is True
                assert os.path.exists(output_path)

                # Check file contents
                with open(output_path) as f:
                    saved_data = json.load(f)
                    assert saved_data == {
                        "nodes": {"A": "Node A", "B": "Node B"},
                        "edges": [["A", "B"]],
                    }

    def test_update_existing_file(self):
        """Test updating an existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.json")

            # Create an existing file
            existing_data = {"nodes": {"A": "Node A"}, "edges": []}
            with open(output_path, "w") as f:
                json.dump(existing_data, f)

            with (
                mock.patch("graph_of_thoughts.graph_utils.console.log"),
                mock.patch("graph_of_thoughts.graph_utils.parse_chain_of_thought") as mock_parse,
            ):
                # Create a valid chain with new data
                mock_parse.return_value = ChainOfThought(nodes={"B": "Node B"}, edges=[["A", "B"]])

                # Call the function
                result = update_and_save_graph(None, output_path, "valid")

                # Check result
                assert result is True

                # Check file contents
                with open(output_path) as f:
                    saved_data = json.load(f)
                    assert saved_data == {
                        "nodes": {"A": "Node A", "B": "Node B"},
                        "edges": [["A", "B"]],
                    }

    def test_with_debug_path(self):
        """Test with a debug path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.json")
            debug_path = os.path.join(temp_dir, "debug.txt")

            with (
                mock.patch("graph_of_thoughts.graph_utils.console.log"),
                mock.patch("graph_of_thoughts.graph_utils.parse_chain_of_thought") as mock_parse,
            ):
                # Create a valid chain
                mock_parse.return_value = ChainOfThought(nodes={"A": "Node A"}, edges=[])

                # Call the function
                result = update_and_save_graph(None, output_path, "raw output", debug_path)

                # Check result
                assert result is True
                assert os.path.exists(debug_path)

                # Check debug file contents
                with open(debug_path) as f:
                    debug_content = f.read()
                    assert debug_content == "raw output"


class TestLoadGraph:
    def test_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with mock.patch("graph_of_thoughts.graph_utils.Path.resolve") as mock_resolve:
            mock_path = mock.MagicMock()
            mock_path.exists.return_value = False
            mock_resolve.return_value = mock_path

            result = load_graph("nonexistent.json")
            assert result == {"nodes": {}, "edges": []}

    def test_valid_file(self):
        """Test loading a valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "graph.json")

            # Create a valid file
            test_data = {"nodes": {"A": "Node A"}, "edges": [["A", "B"]]}
            with open(file_path, "w") as f:
                json.dump(test_data, f)

            # Load the file
            result = load_graph(file_path)
            assert result == test_data

    def test_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "invalid.json")

            # Create an invalid file
            with open(file_path, "w") as f:
                f.write("invalid json")

            # Load the file
            with mock.patch("graph_of_thoughts.graph_utils.console.log"):
                result = load_graph(file_path)
                assert result == {"nodes": {}, "edges": []}
