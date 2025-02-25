# tests/test_integration_llm_graph.py

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from graph_of_thoughts.evaluate_llm_graph import (
    GraphMetrics,
    KnowledgeGraph,
    generate_report,
    load_graph,
)
from tests.fixtures.graph_data import (
    BASELINE_GRAPH,
    GENERATED_GRAPH,
    TINY_BASELINE_GRAPH,
    TINY_GENERATED_GRAPH,
)


class TestIntegrationLLMGraph(unittest.TestCase):
    """Integration tests for LLM graph evaluation components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Define paths for test files
        self.baseline_path = self.temp_path / "baseline.json"
        self.generated_path = self.temp_path / "generated.json"
        self.report_path = self.temp_path / "report.txt"
        self.baseline_img_path = self.temp_path / "baseline.png"
        self.generated_img_path = self.temp_path / "generated.png"

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_end_to_end_tiny_graphs(self):
        """Test the entire graph evaluation pipeline with tiny graphs."""
        # Write the graph files
        with open(self.baseline_path, "w") as f:
            json.dump(TINY_BASELINE_GRAPH, f)

        with open(self.generated_path, "w") as f:
            json.dump(TINY_GENERATED_GRAPH, f)

        # Load the graphs
        baseline_data = load_graph(self.baseline_path)
        generated_data = load_graph(self.generated_path)

        # Create knowledge graph objects
        baseline_kg = KnowledgeGraph("Baseline")
        baseline_kg.add_nodes_and_edges(baseline_data["nodes"], baseline_data["edges"])

        generated_kg = KnowledgeGraph("Generated")
        generated_kg.add_nodes_and_edges(generated_data["nodes"], generated_data["edges"])

        # Skip the actual visualization to avoid matplotlib dependency in tests
        with (
            patch("matplotlib.pyplot.figure"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch("networkx.draw"),
            patch("networkx.draw_networkx_labels"),
            patch("networkx.spring_layout"),
            patch("builtins.print"),
        ):
            baseline_kg.visualize_graph(self.baseline_img_path)
            generated_kg.visualize_graph(self.generated_img_path)

        # Skip the actual embedding model for metrics calculation
        with patch("graph_of_thoughts.evaluate_llm_graph.GraphMatcher") as mock_matcher_cls:
            # Configure the mock matcher to match only the A->B edge
            mock_matcher = mock_matcher_cls.return_value
            mock_matcher.match_edges.return_value = {("A", "B")}

            # Calculate metrics
            metrics = GraphMetrics(baseline_data, generated_data).compute_metrics()

            # Check some key metrics
            self.assertEqual(
                metrics["nodes"]["precision"], 2 / 3
            )  # 2 true positives, 1 false positive
            self.assertEqual(
                metrics["nodes"]["recall"], 2 / 3
            )  # 2 true positives, 1 false negative
            self.assertEqual(metrics["missing_nodes"], ["C"])
            self.assertEqual(metrics["extra_nodes"], ["D"])

        # Test report generation
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("graph_of_thoughts.evaluate_llm_graph.REPORT_PATH", self.report_path),
            patch("builtins.print"),
        ):
            generate_report(metrics)
            mock_file.assert_called_once_with(self.report_path, "w")

    @patch("graph_of_thoughts.evaluate_llm_graph.GraphMatcher")
    def test_graph_save_and_load(self, mock_matcher_cls):
        """Test saving graphs to JSON and loading them back."""
        # Configure mock matcher
        mock_matcher = mock_matcher_cls.return_value
        # We'll assume 3 edges match based on semantic similarity
        mock_matcher.match_edges.return_value = {
            ("Machine Learning", "Supervised Learning"),
            ("Machine Learning", "Reinforcement Learning"),
            ("Supervised Learning", "Neural Networks"),
        }

        # Create and save baseline graph
        baseline_kg = KnowledgeGraph("Baseline")
        baseline_kg.add_nodes_and_edges(BASELINE_GRAPH["nodes"], BASELINE_GRAPH["edges"])

        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json,
            patch("builtins.print"),
        ):
            baseline_kg.save_graph_json(self.baseline_path)
            mock_file.assert_called_once_with(self.baseline_path, "w")
            mock_json.assert_called_once()

        # Create and save generated graph
        generated_kg = KnowledgeGraph("Generated")
        generated_kg.add_nodes_and_edges(GENERATED_GRAPH["nodes"], GENERATED_GRAPH["edges"])

        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json,
            patch("builtins.print"),
        ):
            generated_kg.save_graph_json(self.generated_path)
            mock_file.assert_called_once_with(self.generated_path, "w")
            mock_json.assert_called_once()

        # Test metrics calculation with the fixture data
        metrics = GraphMetrics(BASELINE_GRAPH, GENERATED_GRAPH).compute_metrics()

        # Check if the metrics calculation handles complex graphs correctly
        self.assertTrue(0 <= metrics["nodes"]["precision"] <= 1)
        self.assertTrue(0 <= metrics["nodes"]["recall"] <= 1)
        self.assertTrue(0 <= metrics["edges"]["precision"] <= 1)
        self.assertTrue(0 <= metrics["edges"]["recall"] <= 1)

        # Verify extra and missing elements are identified
        self.assertIn("Deep Learning", metrics["extra_nodes"])
        self.assertIn("Unsupervised Learning", metrics["missing_nodes"])


if __name__ == "__main__":
    unittest.main()
