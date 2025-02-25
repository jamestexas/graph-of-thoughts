import networkx as nx
import json
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "output"
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_graph.json")
LLM_PATH = os.path.join(OUTPUT_DIR, "llm_graph.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "graph_comparison.txt")

# Load a sentence transformer for semantic similarity
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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

    def _get_graph_nodes(self):
        result = {}
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            if "data" in node_data and "content" in node_data["data"]:
                result[node] = node_data["data"]["content"]
            else:
                result[node] = "No description"

        return result

    def save_graph_json(self, filename):
        """Save graph structure as JSON"""
        graph_data = dict(
            name=self.name,
            nodes=self._get_graph_nodes(),
            edges=list(self.graph.edges),
        )
        # nodes={
        #     n: self.graph.nodes[n]["data"]["content"]
        #     if "data" in self.graph.nodes[n] and "content" in self.graph.nodes[n]["data"]
        #     else "No description" for n in self.graph.nodes},
        # edges=list(self.graph.edges),
        # }
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=2)
        print(f"✅ Graph saved to {filename}")

    def visualize_graph(self, filename):
        """Generate a visual representation of the graph"""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos, with_labels=True, node_size=2000, node_color="lightblue"
        )
        labels = {node: self.graph.nodes[node]["desc"] for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)
        plt.title(f"Knowledge Graph: {self.name}")
        plt.savefig(filename)
        plt.close()
        print(f"📸 Graph visualization saved as {filename}")


def load_graph(file_path):
    """Load graph data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def compute_semantic_similarity(text1, text2, threshold=0.75):
    """Check if two texts are semantically similar using embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = (embeddings[0] @ embeddings[1]) / (
        sum(embeddings[0] ** 2) ** 0.5 * sum(embeddings[1] ** 2) ** 0.5
    )
    return similarity > threshold


def match_edges(baseline_edges, generated_edges):
    """Find semantically similar edges."""
    matched_edges = set()
    for base_edge in baseline_edges:
        for gen_edge in generated_edges:
            if compute_semantic_similarity(" → ".join(base_edge), " → ".join(gen_edge)):
                matched_edges.add(base_edge)
                break
    return matched_edges


def compute_graph_metrics(baseline, generated):
    """Compute precision, recall, and F1-score for nodes and edges."""
    baseline_nodes = set(baseline["nodes"].keys())
    generated_nodes = set(generated["nodes"].keys())

    baseline_edges = set(tuple(e) for e in baseline["edges"])
    generated_edges = set(tuple(e) for e in generated["edges"])

    # Node metrics
    true_positive_nodes = len(baseline_nodes & generated_nodes)
    false_positive_nodes = len(generated_nodes - baseline_nodes)
    false_negative_nodes = len(baseline_nodes - generated_nodes)

    precision_nodes = (
        true_positive_nodes / (true_positive_nodes + false_positive_nodes)
        if (true_positive_nodes + false_positive_nodes) > 0
        else 0
    )
    recall_nodes = (
        true_positive_nodes / (true_positive_nodes + false_negative_nodes)
        if (true_positive_nodes + false_negative_nodes) > 0
        else 0
    )
    f1_nodes = (
        2 * (precision_nodes * recall_nodes) / (precision_nodes + recall_nodes)
        if (precision_nodes + recall_nodes) > 0
        else 0
    )

    # Edge metrics (semantic matching)
    matched_edges = match_edges(baseline_edges, generated_edges)
    true_positive_edges = len(matched_edges)
    false_positive_edges = len(generated_edges - matched_edges)
    false_negative_edges = len(baseline_edges - matched_edges)

    precision_edges = (
        true_positive_edges / (true_positive_edges + false_positive_edges)
        if (true_positive_edges + false_positive_edges) > 0
        else 0
    )
    recall_edges = (
        true_positive_edges / (true_positive_edges + false_negative_edges)
        if (true_positive_edges + false_negative_edges) > 0
        else 0
    )
    f1_edges = (
        2 * (precision_edges * recall_edges) / (precision_edges + recall_edges)
        if (precision_edges + recall_edges) > 0
        else 0
    )

    return {
        "nodes": {"precision": precision_nodes, "recall": recall_nodes, "f1": f1_nodes},
        "edges": {"precision": precision_edges, "recall": recall_edges, "f1": f1_edges},
        "extra_nodes": generated_nodes - baseline_nodes,
        "missing_nodes": baseline_nodes - generated_nodes,
        "extra_edges": generated_edges - baseline_edges,
        "missing_edges": baseline_edges - generated_edges,
    }


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
    baseline_graph_data = load_graph(BASELINE_PATH)
    llm_graph_data = load_graph(LLM_PATH)

    # Compute metrics
    metrics = compute_graph_metrics(baseline_graph_data, llm_graph_data)

    # Visualize graphs
    KnowledgeGraph("Baseline").visualize_graph(
        os.path.join(OUTPUT_DIR, "baseline_graph.png")
    )
    KnowledgeGraph("LLM Output").visualize_graph(
        os.path.join(OUTPUT_DIR, "llm_graph.png")
    )

    # Generate report
    generate_report(metrics)


if __name__ == "__main__":
    main()
