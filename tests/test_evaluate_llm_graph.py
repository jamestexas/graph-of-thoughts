# tests/test_evaluate_llm_graph.py

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import networkx as nx

from graph_of_thoughts.evaluate_llm_graph import (
    GraphMatcher,
    GraphMetrics,
    KnowledgeGraph,
    generate_report,
    load_graph,
)


class TestLoadGraph(unittest.TestCase):
    """Test the load_graph function."""

    def test_load_graph_file_exists(self):
        """Test loading a graph from an existing file."""
        test_data = {
            "name": "Test Graph",
            "nodes": {"A": "Node A", "B": "Node B"},
            "edges": [["A", "B"]],
        }

        # Mock the Path.read_text directly instead of open
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=json.dumps(test_data)),
        ):
            result = load_graph("dummy_path.json")

            # Check the loaded data
            self.assertEqual(result, test_data)

    def test_load_graph_file_not_exists(self):
        """Test loading a graph when the file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = load_graph("nonexistent_path.json")

            # Should return empty graph data
            self.assertEqual(result, {"nodes": {}, "edges": []})


class TestKnowledgeGraph(unittest.TestCase):
    """Test the KnowledgeGraph class."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = KnowledgeGraph("Test Graph")

        # Sample graph data
        self.nodes = {
            "A": "Node A description",
            "B": "Node B description",
            "C": "Node C description",
        }
        self.edges = [["A", "B"], ["B", "C"]]

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.graph.name, "Test Graph")
        self.assertIsInstance(self.graph.graph, nx.DiGraph)
        self.assertEqual(len(self.graph.graph.nodes), 0)
        self.assertEqual(len(self.graph.graph.edges), 0)

    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges to the graph."""
        self.graph.add_nodes_and_edges(self.nodes, self.edges)

        # Check that nodes were added with correct descriptions
        self.assertEqual(len(self.graph.graph.nodes), 3)
        for node, desc in self.nodes.items():
            self.assertIn(node, self.graph.graph.nodes)
            self.assertEqual(self.graph.graph.nodes[node]["desc"], desc)

        # Check that edges were added
        self.assertEqual(len(self.graph.graph.edges), 2)
        for source, target in self.edges:
            self.assertIn((source, target), self.graph.graph.edges)

    def test_add_nodes_and_edges_with_invalid_edges(self):
        """Test adding edges with nodes that don't exist."""
        # Edges that refer to non-existent nodes
        invalid_edges = [["A", "Z"], ["X", "Y"]]

        self.graph.add_nodes_and_edges(self.nodes, invalid_edges)

        # Should add nodes but skip invalid edges
        self.assertEqual(len(self.graph.graph.nodes), 3)
        self.assertEqual(len(self.graph.graph.edges), 0)

    def test_save_graph_json(self):
        """Test saving graph to JSON file."""
        self.graph.add_nodes_and_edges(self.nodes, self.edges)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Test with a mock to avoid actual file I/O in tests
            with (
                patch("builtins.open", mock_open()) as mock_file,
                patch("json.dump") as mock_json,
                patch("builtins.print") as mock_print,
            ):
                self.graph.save_graph_json(temp_path)

                # Check that file was opened and JSON was dumped
                mock_file.assert_called_once_with(temp_path, "w")
                mock_json.assert_called_once()
                mock_print.assert_called_once()

                # Verify the structure of the data being saved
                saved_data = mock_json.call_args[0][0]
                self.assertEqual(saved_data["name"], "Test Graph")
                self.assertEqual(len(saved_data["nodes"]), 3)
                self.assertEqual(len(saved_data["edges"]), 2)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_graph_nodes(self):
        """Test the _get_graph_nodes method."""
        # Add nodes with data structure mimicking the actual format
        graph = nx.DiGraph()
        graph.add_node("node1", data={"content": "Content 1"})
        graph.add_node("node2", data={})  # Missing content
        graph.add_node("node3")  # Missing data attribute

        self.graph.graph = graph

        # Call the method
        result = self.graph._get_graph_nodes()

        # Check results
        self.assertEqual(result["node1"], "Content 1")
        self.assertEqual(result["node2"], "No description")
        self.assertEqual(result["node3"], "No description")

    @patch("matplotlib.pyplot.figure")
    @patch("networkx.spring_layout")
    @patch("networkx.draw")
    @patch("networkx.draw_networkx_labels")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("builtins.print")
    def test_visualize_graph(
        self,
        mock_print,
        mock_close,
        mock_savefig,
        mock_title,
        mock_draw_labels,
        mock_draw,
        mock_layout,
        mock_figure,
    ):
        """Test graph visualization."""
        self.graph.add_nodes_and_edges(self.nodes, self.edges)

        # Call the method
        self.graph.visualize_graph("test_graph.png")

        # Check that all matplotlib functions were called
        mock_figure.assert_called_once()
        mock_layout.assert_called_once()
        mock_draw.assert_called_once()
        mock_draw_labels.assert_called_once()
        mock_title.assert_called_once()
        mock_savefig.assert_called_once_with("test_graph.png")
        mock_close.assert_called_once()
        mock_print.assert_called_once()


class TestGraphMatcher(unittest.TestCase):
    """Test the GraphMatcher class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the SentenceTransformer
        self.mock_model = MagicMock()
        self.graph_matcher = GraphMatcher(model=self.mock_model)

    def test_compute_semantic_similarity_above_threshold(self):
        """Test semantic similarity computation when above threshold."""
        # Mock embeddings that would give high similarity
        self.mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],  # First embedding
            [0.9, 0.1, 0.0],  # Second embedding (close to first)
        ]

        result = self.graph_matcher.compute_semantic_similarity("text1", "text2", threshold=0.8)

        # Should be similar (above threshold)
        self.assertTrue(result)
        self.mock_model.encode.assert_called_once_with(["text1", "text2"])

    def test_compute_semantic_similarity_below_threshold(self):
        """Test semantic similarity computation when below threshold."""
        # Mock embeddings that would give low similarity
        self.mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],  # First embedding
            [0.0, 1.0, 0.0],  # Second embedding (orthogonal to first)
        ]

        result = self.graph_matcher.compute_semantic_similarity("text1", "text2", threshold=0.5)

        # Should not be similar (below threshold)
        self.assertFalse(result)

    def test_match_edges(self):
        """Test the edge matching functionality."""
        # Setup mock for compute_semantic_similarity
        with patch.object(self.graph_matcher, "compute_semantic_similarity") as mock_sim:
            # Configure which edges should match
            mock_sim.side_effect = lambda a, b, threshold=0.75: a == "A → B" and b == "X → Y"

            baseline_edges = [("A", "B"), ("C", "D")]
            generated_edges = [("X", "Y"), ("P", "Q")]

            matched = self.graph_matcher.match_edges(baseline_edges, generated_edges)

            # Only ("A", "B") should match with ("X", "Y")
            self.assertEqual(matched, {("A", "B")})

            # Check that all combinations were compared
            self.assertEqual(mock_sim.call_count, 4)  # 2x2 comparisons


class TestGraphMetrics(unittest.TestCase):
    """Test the GraphMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample graph data
        self.baseline = {
            "nodes": {"A": "Node A", "B": "Node B", "C": "Node C"},
            "edges": [["A", "B"], ["B", "C"]],
        }

        self.generated = {
            "nodes": {"A": "Node A", "B": "Node B", "D": "Node D"},
            "edges": [["A", "B"], ["B", "D"]],
        }

        # Mock the GraphMatcher
        with patch("graph_of_thoughts.evaluate_llm_graph.GraphMatcher") as mock_matcher_cls:
            mock_matcher = mock_matcher_cls.return_value
            # Configure the mock matcher to match certain edges
            mock_matcher.match_edges.return_value = {("A", "B")}

            self.graph_metrics = GraphMetrics(self.baseline, self.generated)

            # Store the mock for later assertions
            self.mock_matcher = mock_matcher

    def test_compute_metrics(self):
        """Test the metrics computation."""
        metrics = self.graph_metrics.compute_metrics()

        # Check node metrics
        self.assertEqual(metrics["nodes"]["precision"], 2 / 3)  # TP=2, FP=1
        self.assertEqual(metrics["nodes"]["recall"], 2 / 3)  # TP=2, FN=1
        self.assertEqual(metrics["nodes"]["f1"], 2 / 3)  # Harmonic mean of precision and recall

        # Check edge metrics
        self.assertEqual(metrics["edges"]["precision"], 1 / 2)  # TP=1, FP=1
        self.assertEqual(metrics["edges"]["recall"], 1 / 2)  # TP=1, FN=1
        self.assertEqual(metrics["edges"]["f1"], 1 / 2)  # Harmonic mean of precision and recall

        # Check extra/missing nodes and edges
        self.assertEqual(metrics["extra_nodes"], ["D"])
        self.assertEqual(metrics["missing_nodes"], ["C"])
        self.assertEqual(metrics["extra_edges"], [["B", "D"]])
        self.assertEqual(metrics["missing_edges"], [["B", "C"]])

        # Verify the matcher was used correctly
        self.mock_matcher.match_edges.assert_called_once()

    def test_calculate_precision(self):
        """Test precision calculation."""
        self.assertEqual(GraphMetrics._calculate_precision(3, 1), 0.75)  # TP=3, FP=1
        self.assertEqual(GraphMetrics._calculate_precision(0, 0), 0)  # Handle division by zero

    def test_calculate_recall(self):
        """Test recall calculation."""
        self.assertEqual(GraphMetrics._calculate_recall(3, 1), 0.75)  # TP=3, FN=1
        self.assertEqual(GraphMetrics._calculate_recall(0, 0), 0)  # Handle division by zero

    def test_calculate_f1(self):
        """Test F1 score calculation."""
        self.assertEqual(GraphMetrics._calculate_f1(0.8, 0.6), 0.6857142857142857)  # 2*P*R/(P+R)
        self.assertEqual(GraphMetrics._calculate_f1(0, 0), 0)  # Handle division by zero


class TestGenerateReport(unittest.TestCase):
    """Test the generate_report function."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = {
            "nodes": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "edges": {"precision": 0.6, "recall": 0.5, "f1": 0.55},
            "extra_nodes": ["D", "E"],
            "missing_nodes": ["C"],
            "extra_edges": [["A", "D"]],
            "missing_edges": [["B", "C"]],
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    @patch("graph_of_thoughts.evaluate_llm_graph.REPORT_PATH", Path("test_report.txt"))
    def test_generate_report(self, mock_print, mock_file):
        """Test report generation."""
        generate_report(self.metrics)

        # Check that file was opened and written to
        mock_file.assert_called_once_with(Path("test_report.txt"), "w")
        mock_file().write.assert_called_once()

        # Check that the report was printed
        self.assertEqual(mock_print.call_count, 1)

        # Check that the report contains all the metrics
        written_content = mock_file().write.call_args[0][0]
        self.assertIn("Extra Nodes: ['D', 'E']", written_content)
        self.assertIn("Missing Nodes: ['C']", written_content)
        self.assertIn("Node Precision: 0.80", written_content)
        self.assertIn("Edge F1 Score: 0.55", written_content)


if __name__ == "__main__":
    unittest.main()
