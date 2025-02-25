import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer

from graph_of_thoughts.constants import EMBEDDING_MODEL, LLM_PATH, OUTPUT_DIR, console
from graph_of_thoughts.utils import get_sentence_transformer

# Load a sentence transformer for semantic similarity
BASELINE_PATH: Path = OUTPUT_DIR / "baseline_graph.json"
REPORT_PATH: Path = OUTPUT_DIR / "graph_comparison.txt"


def load_graph(file_path: str | Path) -> dict:
    """Load graph data from a JSON file."""
    p = Path(file_path)
    if p.exists():
        return json.loads(p.read_text())
    return dict(nodes={}, edges=[])


class KnowledgeGraph:
    def __init__(self, name="Generated"):
        self.graph = nx.DiGraph()
        self.name = name

    def add_nodes_and_edges(self, nodes: dict[str, str], edges: list[list[str]]):
        for node, desc in nodes.items():
            self.graph.add_node(node, desc=desc)
        for source, target in edges:
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target)

    def save_graph_json(self, filename: str | Path) -> None:
        """Save graph structure as JSON"""
        graph_data = dict(
            name=self.name,
            nodes={
                node: self.graph.nodes[node].get("desc", "No description")
                for node in self.graph.nodes
            },
            edges=list(self.graph.edges),
        )
        try:
            with open(filename, "w") as f:
                json.dump(graph_data, f, indent=2)
        except Exception as e:
            console.log(f"Error saving graph to JSON: {e}")
        else:
            print(f"✅ Graph saved to {filename}")

    def _get_graph_nodes(self):
        result = {}
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            if "data" in node_data and "content" in node_data["data"]:
                result[node] = node_data["data"]["content"]
            else:
                result[node] = "No description"

        return result

    def visualize_graph(self, filename):
        """Generate a visual representation of the graph"""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
        )
        labels = {node: self.graph.nodes[node]["desc"] for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)
        plt.title(f"Knowledge Graph: {self.name}")
        plt.savefig(filename)
        plt.close()
        print(f"📸 Graph visualization saved as {filename}")


class GraphMatcher:
    """Handles semantic similarity and edge matching between graphs."""

    def __init__(
        self,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.model = model or get_sentence_transformer(model_name=EMBEDDING_MODEL)

    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.75,
    ):
        """Check if two texts are semantically similar using embeddings."""
        embeddings = self.model.encode([text1, text2])
        similarity = (embeddings[0] @ embeddings[1]) / (
            sum(embeddings[0] ** 2) ** 0.5 * sum(embeddings[1] ** 2) ** 0.5
        )
        return similarity > threshold

    def match_edges(self, baseline_edges, generated_edges):
        """Find semantically similar edges."""
        matched_edges = set()
        for base_edge in baseline_edges:
            for gen_edge in generated_edges:
                if self.compute_semantic_similarity(" → ".join(base_edge), " → ".join(gen_edge)):
                    matched_edges.add(base_edge)
                    break
        return matched_edges


class GraphMetrics:
    """Computes precision, recall, and F1-score for graph comparison."""

    def __init__(self, baseline, generated):
        self.baseline = baseline
        self.generated = generated
        self.matcher = GraphMatcher()

    def compute_metrics(self):
        """Compute precision, recall, and F1-score for nodes and edges."""
        baseline_nodes = set(self.baseline["nodes"].keys())
        generated_nodes = set(self.generated["nodes"].keys())

        baseline_edges = set(tuple(e) for e in self.baseline["edges"])
        generated_edges = set(tuple(e) for e in self.generated["edges"])

        # Node metrics
        true_positive_nodes = len(baseline_nodes & generated_nodes)
        false_positive_nodes = len(generated_nodes - baseline_nodes)
        false_negative_nodes = len(baseline_nodes - generated_nodes)

        precision_nodes = self._calculate_precision(true_positive_nodes, false_positive_nodes)
        recall_nodes = self._calculate_recall(true_positive_nodes, false_negative_nodes)
        f1_nodes = self._calculate_f1(precision_nodes, recall_nodes)

        # Edge metrics (semantic matching)
        matched_edges = self.matcher.match_edges(baseline_edges, generated_edges)
        true_positive_edges = len(matched_edges)
        false_positive_edges = len(generated_edges - matched_edges)
        false_negative_edges = len(baseline_edges - matched_edges)

        precision_edges = self._calculate_precision(true_positive_edges, false_positive_edges)
        recall_edges = self._calculate_recall(true_positive_edges, false_negative_edges)
        f1_edges = self._calculate_f1(precision_edges, recall_edges)

        return dict(
            nodes=dict(
                precision=precision_nodes,
                recall=recall_nodes,
                f1=f1_nodes,
            ),
            edges=dict(
                precision=precision_edges,
                recall=recall_edges,
                f1=f1_edges,
            ),
            extra_nodes=sorted(generated_nodes - baseline_nodes),
            missing_nodes=sorted(baseline_nodes - generated_nodes),
            extra_edges=sorted(generated_edges - baseline_edges),
            missing_edges=sorted(baseline_edges - generated_edges),
        )

    @staticmethod
    def _calculate_precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @staticmethod
    def _calculate_recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def _calculate_f1(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def generate_report(metrics):
    """Generate and save a structured comparison report."""
    report = [
        "📌 **Graph Comparison Report**",
        f"🟢 Extra Nodes: {metrics['extra_nodes']}",
        f"🔴 Missing Nodes: {metrics['missing_nodes']}",
        f"🟢 Extra Edges: {metrics['extra_edges']}",
        f"🔴 Missing Edges: {metrics['missing_edges']}",
        "\n📊 **Performance Metrics**",
        f"🔹 Node Precision: {metrics['nodes']['precision']:.2f}",
        f"🔹 Node Recall: {metrics['nodes']['recall']:.2f}",
        f"🔹 Node F1 Score: {metrics['nodes']['f1']:.2f}",
        f"🔹 Edge Precision: {metrics['edges']['precision']:.2f}",
        f"🔹 Edge Recall: {metrics['edges']['recall']:.2f}",
        f"🔹 Edge F1 Score: {metrics['edges']['f1']:.2f}",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))

    print("\n".join(report))


def main():
    """Run the graph comparison and evaluation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load graphs
    baseline_graph_data = KnowledgeGraph("Baseline")
    generated_graph_data = KnowledgeGraph("LLM Output")

    baseline_graph_data.add_nodes_and_edges(*load_graph(BASELINE_PATH))
    generated_graph_data.add_nodes_and_edges(*load_graph(LLM_PATH))

    # Compute metrics
    metrics = GraphMetrics(baseline_graph_data, generated_graph_data).compute_metrics()

    # Visualize graphs
    baseline_graph_data.visualize_graph(os.path.join(OUTPUT_DIR, "baseline_graph.png"))
    generated_graph_data.visualize_graph(os.path.join(OUTPUT_DIR, "llm_graph.png"))

    # Generate report
    generate_report(metrics)


if __name__ == "__main__":
    main()
