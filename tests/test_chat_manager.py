# tests/test_chat_manager.py

import unittest

from unittest.mock import MagicMock, patch
import networkx as nx

from pathlib import Path
import os

# Import the components to test
from graph_of_thoughts.chat_manager import ChatManager
from graph_of_thoughts.context_manager import ContextGraphManager
from graph_of_thoughts.models import SeedData


class TestChatManager(unittest.TestCase):
    def get_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer_mock = MagicMock()

        # Create a mock dict-like object that also has a to() method
        tokenizer_return = MagicMock()
        tokenizer_return.__getitem__ = lambda self, key: MagicMock()
        tokenizer_return.to = MagicMock(return_value=tokenizer_return)
        tokenizer_mock.return_value = tokenizer_return
        tokenizer_mock.decode.return_value = "Mock LLM response"
        tokenizer_mock.eos_token_id = 0

        return tokenizer_mock

    def setUp(self):
        """Setup test fixtures before each test method."""
        # TODO: Make this a method too so init is smol
        # Create a mock context manager
        self.context_manager = MagicMock(spec=ContextGraphManager)

        # Mock the graph storage
        self.context_manager.graph_storage = MagicMock()
        self.context_manager.graph_storage.graph = nx.DiGraph()

        # Mock methods we'll use
        self.context_manager.query_context.return_value = ["node1", "node2", "node3"]
        self.context_manager.visualize_graph_as_text.return_value = "Mock graph text"

        # Assign the tokenizer as an attribute
        type(self.context_manager).tokenizer = self.get_mock_tokenizer()

        # Initialize the chat manager with our mock
        self.chat_manager = ChatManager(self.context_manager)

        # Create a temporary output directory for tests
        self.temp_output_dir = Path("tests/temp_output")
        os.makedirs(self.temp_output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        for file in self.temp_output_dir.glob("*"):
            try:
                file.unlink()
            except Exception:
                pass
        if self.temp_output_dir.exists():
            try:
                self.temp_output_dir.rmdir()
            except Exception:
                pass

    @patch("graph_of_thoughts.chat_manager.OUTPUT_DIR", Path("tests/temp_output"))
    @patch("graph_of_thoughts.chat_manager.console")
    def test_generate_response(self, mock_console):
        """Test the generate_response method."""
        # Setup
        query = "Test query"
        max_new_tokens = 100

        # Create a mock for the model.generate() call
        model_mock = MagicMock()
        model_mock.generate.return_value = [MagicMock()]  # Mock output tensor
        model_mock.device = "cpu"
        self.context_manager.model = model_mock

        # Call the method
        response = self.chat_manager.generate_response(query, max_new_tokens)

        # Assertions
        self.context_manager.query_context.assert_called_with(query, top_k=3)
        self.context_manager.tokenizer.assert_called_once()
        self.context_manager.model.generate.assert_called_once()
        self.context_manager.tokenizer.decode.assert_called_once()
        self.assertEqual(response, "Mock LLM response")

    @patch("graph_of_thoughts.chat_manager.extract_and_clean_json")
    @patch("graph_of_thoughts.chat_manager.console")
    def test_process_turn(self, mock_console, mock_extract_json):
        """Test processing a single conversation turn."""
        # Setup
        user_input = "Test user input"
        conversation_turn = 1

        # Configure mocks
        mock_extract_json.return_value = {"nodes": [], "edges": []}

        # Create a mock for the generate_response method to avoid having to mock the full LLM call chain
        self.chat_manager.generate_response = MagicMock(
            return_value="Mock LLM response"
        )

        # Call the method
        response = self.chat_manager.process_turn(user_input, conversation_turn)

        # Assertions
        self.context_manager.decay_importance.assert_called_once()
        self.context_manager.add_context.assert_any_call(
            f"user_{conversation_turn}", user_input
        )
        self.context_manager.add_context.assert_any_call(
            f"llm_{conversation_turn}", response
        )
        self.context_manager.query_context.assert_called_with(user_input, top_k=3)
        self.context_manager.iterative_refinement.assert_called_once()
        self.context_manager.prune_context.assert_called_once()
        mock_console.log.assert_called()
        self.assertEqual(response, "Mock LLM response")

    @patch("graph_of_thoughts.chat_manager.OUTPUT_DIR", Path("tests/temp_output"))
    @patch("graph_of_thoughts.chat_manager.seed_nodes")
    def test_simulate_conversation(self, mock_seed_nodes):
        """Test simulating a full conversation."""
        # Setup
        inputs = ["Question 1", "Question 2"]

        # Create a proper SeedData object with all required fields
        # First, let's create a mock SeedData that matches the expected structure
        mock_seed_data = MagicMock(spec=SeedData)

        # Patch the process_turn method to avoid complex mocking
        with patch.object(
            self.chat_manager, "process_turn", return_value="Mocked response"
        ):
            # Call the method
            results = self.chat_manager.simulate_conversation(
                inputs, [mock_seed_data], "test_experiment"
            )

            # Assertions
            mock_seed_nodes.assert_called_once()
            self.assertEqual(len(results), 2)  # One result per input

            # Check if files were created
            baseline_path = self.temp_output_dir / "test_experiment_baseline_graph.json"
            llm_path = self.temp_output_dir / "test_experiment_llm_graph.json"
            experiment_path = self.temp_output_dir / "test_experiment_data.json"

            self.assertTrue(baseline_path.exists())
            self.assertTrue(llm_path.exists())
            self.assertTrue(experiment_path.exists())

            # Check the experiment data structure
            self.assertIn("turn", results[0])
            self.assertIn("user_input", results[0])
            self.assertIn("llm_response", results[0])
            self.assertIn("retrieved_context", results[0])
            self.assertIn("graph_before", results[0])
            self.assertIn("graph_after", results[0])
            self.assertIn("metrics", results[0])

    def test_extract_nodes(self):
        """Test the _extract_nodes helper method."""
        # Setup
        graph = nx.DiGraph()
        # Node with data
        graph.add_node("node1", data={"content": "Node 1 content"})
        # Node without content
        graph.add_node("node2", data={})
        # Node without data attribute
        graph.add_node("node3")

        # Call the method
        nodes_dict = self.chat_manager._extract_nodes(graph)

        # Assertions
        self.assertEqual(nodes_dict["node1"], "Node 1 content")
        self.assertEqual(nodes_dict["node2"], "No description")
        self.assertEqual(nodes_dict["node3"], "No description")

    def test_serialize_graph(self):
        """Test graph serialization for JSON output."""
        # Setup
        graph = nx.DiGraph()

        # Add a node with a complex object that has model_dump method
        class DummyModel:
            def model_dump(self):
                return {"field1": "value1", "field2": "value2"}

        graph.add_node("node1", complex_data=DummyModel())
        graph.add_node("node2", simple_data="simple value")
        graph.add_edge("node1", "node2")

        # Call the method
        serialized = ChatManager._serialize_graph(graph)

        # Assertions
        self.assertIn("nodes", serialized)
        self.assertIn("links", serialized)

        # Check that model_dump was called
        nodes = serialized["nodes"]
        node1 = next(n for n in nodes if n["id"] == "node1")
        self.assertIsInstance(node1["complex_data"], dict)
        self.assertEqual(node1["complex_data"]["field1"], "value1")

        # Check that simple values are preserved
        node2 = next(n for n in nodes if n["id"] == "node2")
        self.assertEqual(node2["simple_data"], "simple value")

    def test_datetime_handler(self):
        """Test datetime serialization handler."""
        from datetime import datetime

        # Test with datetime
        dt = datetime(2023, 1, 1, 12, 0, 0)
        self.assertEqual(ChatManager._datetime_handler(dt), "2023-01-01T12:00:00")

        # Test with set
        test_set = {1, 2, 3}
        self.assertEqual(sorted(ChatManager._datetime_handler(test_set)), [1, 2, 3])

        # Test with unsupported type
        with self.assertRaises(TypeError):
            ChatManager._datetime_handler(complex(1, 2))


if __name__ == "__main__":
    unittest.main()
