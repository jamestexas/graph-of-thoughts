# tests/fixtures/graph_data.py

"""
Test fixtures for graph evaluation tests.
"""

# Sample baseline graph data
BASELINE_GRAPH = {
    "name": "Baseline Graph",
    "nodes": {
        "Machine Learning": "A field of study that gives computers the ability to learn without being explicitly programmed",
        "Supervised Learning": "Learning with labeled training data",
        "Unsupervised Learning": "Learning from unlabeled data",
        "Reinforcement Learning": "Learning through interaction with an environment",
        "Neural Networks": "Computing systems inspired by biological neural networks",
    },
    "edges": [
        ["Machine Learning", "Supervised Learning"],
        ["Machine Learning", "Unsupervised Learning"],
        ["Machine Learning", "Reinforcement Learning"],
        ["Supervised Learning", "Neural Networks"],
    ],
}

# Sample generated graph data (with some differences from baseline)
GENERATED_GRAPH = {
    "name": "Generated Graph",
    "nodes": {
        "Machine Learning": "A field of AI that enables computers to learn from data",
        "Supervised Learning": "Training models using labeled examples",
        "Reinforcement Learning": "Learning through trial and error with rewards",
        "Neural Networks": "Models inspired by human brain structure",
        "Deep Learning": "Neural networks with multiple layers",
        "Convolutional Networks": "Neural networks specialized for processing grid-like data",
    },
    "edges": [
        ["Machine Learning", "Supervised Learning"],
        ["Machine Learning", "Reinforcement Learning"],
        ["Neural Networks", "Deep Learning"],
        ["Deep Learning", "Convolutional Networks"],
        ["Supervised Learning", "Neural Networks"],
    ],
}

# Expected metrics when comparing the above graphs
EXPECTED_METRICS = {
    "nodes": {
        "precision": 0.667,  # 4 true positive nodes out of 6 generated nodes
        "recall": 0.8,  # 4 true positive nodes out of 5 baseline nodes
        "f1": 0.727,  # F1 score from precision and recall
    },
    "edges": {
        # Assuming semantic similarity finds 3 matching edges
        "precision": 0.6,  # 3 true positive edges out of 5 generated edges
        "recall": 0.75,  # 3 true positive edges out of 4 baseline edges
        "f1": 0.667,  # F1 score from precision and recall
    },
    "extra_nodes": ["Convolutional Networks", "Deep Learning"],
    "missing_nodes": ["Unsupervised Learning"],
    "extra_edges": [
        ["Deep Learning", "Convolutional Networks"],
        ["Neural Networks", "Deep Learning"],
    ],
    "missing_edges": [["Machine Learning", "Unsupervised Learning"]],
}

# A very small graph for quick tests
TINY_BASELINE_GRAPH = {
    "name": "Tiny Baseline",
    "nodes": {"A": "Node A", "B": "Node B", "C": "Node C"},
    "edges": [["A", "B"], ["B", "C"]],
}

TINY_GENERATED_GRAPH = {
    "name": "Tiny Generated",
    "nodes": {"A": "Node A", "B": "Node B", "D": "Node D"},
    "edges": [["A", "B"], ["B", "D"]],
}
