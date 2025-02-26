from datetime import datetime, timezone
from unittest import mock

import networkx as nx
import numpy as np
import pytest

from graph_of_thoughts.graph_components import (
    EmbeddingEngine,
    GraphStorage,
    ReasoningEngine,
    build_initial_graph,
)
from graph_of_thoughts.models import ChainOfThought, ContextNode


class TestBuildInitialGraph:
    def test_build_initial_graph(self):
        """Test the build_initial_graph function creates correct structure."""
        graph = build_initial_graph()

        # Check graph structure
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

        # Check node presence
        assert "root" in graph.nodes
        assert "subA" in graph.nodes
        assert "subB" in graph.nodes

        # Check edge connections
        assert graph.has_edge("root", "subA")
        assert graph.has_edge("root", "subB")

        # Check node attributes
        root_data = graph.nodes["root"]["data"]
        assert isinstance(root_data, ContextNode)
        assert root_data.node_id == "root"
        assert "Top-level concept" in root_data.content
        assert root_data.metadata["importance"] == 1.0


class TestGraphStorage:
    @pytest.fixture
    def empty_graph_storage(self):
        """Fixture for an empty graph storage."""
        return GraphStorage()

    @pytest.fixture
    def populated_graph_storage(self):
        """Fixture for a graph storage with pre-populated nodes."""
        graph = nx.DiGraph()

        # Add nodes
        node1_data = {
            "content": "Node 1 content",
            "importance": 1.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        node2_data = {
            "content": "Node 2 content",
            "importance": 0.9,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        graph.add_node("node1", data=node1_data)
        graph.add_node("node2", data=node2_data)

        # Add edge
        graph.add_edge("node1", "node2")

        return GraphStorage(initial_graph=graph)

    def test_initialization_empty(self, empty_graph_storage):
        """Test initialization with an empty graph."""
        assert isinstance(empty_graph_storage.graph, nx.DiGraph)
        assert len(empty_graph_storage.graph.nodes) == 0

    def test_initialization_with_graph(self, populated_graph_storage):
        """Test initialization with an existing graph."""
        assert isinstance(populated_graph_storage.graph, nx.DiGraph)
        assert len(populated_graph_storage.graph.nodes) == 2
        assert populated_graph_storage.graph.has_edge("node1", "node2")

    def test_add_node(self, empty_graph_storage):
        """Test adding a node to the graph."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Add node with default metadata
            empty_graph_storage.add_node("test_node", "Test content")

            # Check node was added
            assert "test_node" in empty_graph_storage.graph.nodes
            node_data = empty_graph_storage.graph.nodes["test_node"]["data"]
            assert node_data["content"] == "Test content"
            assert node_data["importance"] == 1.0

            # Add node with custom metadata
            custom_metadata = {"importance": 0.5, "custom_key": "custom_value"}
            empty_graph_storage.add_node(
                "custom_node", "Custom content", custom_metadata
            )

            # Check node was added with custom metadata
            assert "custom_node" in empty_graph_storage.graph.nodes
            custom_node_data = empty_graph_storage.graph.nodes["custom_node"]["data"]
            assert custom_node_data["content"] == "Custom content"
            assert custom_node_data["importance"] == 0.5

    def test_add_edge(self, populated_graph_storage):
        """Test adding an edge between nodes."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Add a new node
            populated_graph_storage.add_node("node3", "Node 3 content")

            # Add edge between existing nodes
            populated_graph_storage.add_edge("node2", "node3")
            assert populated_graph_storage.graph.has_edge("node2", "node3")

            # Attempt to add edge with non-existent node
            populated_graph_storage.add_edge("node3", "nonexistent")
            assert not populated_graph_storage.graph.has_edge("node3", "nonexistent")

    def test_remove_node(self, populated_graph_storage):
        """Test removing a node and its edges."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Remove existing node
            populated_graph_storage.remove_node("node2")
            assert "node2" not in populated_graph_storage.graph.nodes
            assert not populated_graph_storage.graph.has_edge("node1", "node2")

            # Remove non-existent node (should not raise error)
            populated_graph_storage.remove_node("nonexistent")

    def test_get_node_content(self, populated_graph_storage):
        """Test retrieving node content."""
        # Get content from existing node
        content = populated_graph_storage.get_node_content("node1")
        assert content == "Node 1 content"

        # Get content from non-existent node
        content = populated_graph_storage.get_node_content("nonexistent")
        assert content is None

    def test_to_json(self, populated_graph_storage):
        """Test converting graph to JSON format."""
        # Explicitly set edges parameter to avoid FutureWarning
        with mock.patch("networkx.node_link_data") as mock_node_link_data:
            mock_node_link_data.return_value = {
                "nodes": [
                    {"id": "node1", "data": {"content": "Node 1 content"}},
                    {"id": "node2", "data": {"content": "Node 2 content"}},
                ],
                "links": [{"source": "node1", "target": "node2"}],
            }

            json_data = populated_graph_storage.to_json()

            # Verify method was called with correct parameters
            mock_node_link_data.assert_called_once()

            # Check JSON structure
            assert isinstance(json_data, dict)
            assert "nodes" in json_data
            assert "links" in json_data
            assert len(json_data["nodes"]) == 2
            assert len(json_data["links"]) == 1

    def test_visualize_as_text(self, populated_graph_storage):
        """Test text visualization of the graph."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            text = populated_graph_storage.visualize_as_text()

            # Check text output
            assert isinstance(text, str)
            assert "**Current Knowledge Graph**" in text
            assert "**Nodes**" in text
            assert "**Edges**" in text
            assert "node1" in text
            assert "node2" in text
            assert "Node 1 content" in text
            assert "node1 → node2" in text

    def test_decay_node_importance(self, populated_graph_storage):
        """Test decaying a node's importance."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Get initial importance
            initial_importance = populated_graph_storage.graph.nodes["node1"]["data"][
                "importance"
            ]

            # Apply decay
            populated_graph_storage.decay_node_importance("node1", decay_factor=0.9)

            # Check importance was reduced
            new_importance = populated_graph_storage.graph.nodes["node1"]["data"][
                "importance"
            ]
            assert new_importance < initial_importance

            # Test with non-existent node (should not raise error)
            populated_graph_storage.decay_node_importance("nonexistent")

    def test_prune_low_importance_nodes(self, populated_graph_storage):
        """Test pruning low importance nodes."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Lower node2's importance
            populated_graph_storage.graph.nodes["node2"]["data"]["importance"] = 0.1

            # Prune with threshold higher than node2's importance
            populated_graph_storage.prune_low_importance_nodes(threshold=0.5)

            # Check node2 was removed
            assert "node2" not in populated_graph_storage.graph.nodes

            # Check node1, with higher importance, was kept
            assert "node1" in populated_graph_storage.graph.nodes


class TestEmbeddingEngine:
    # In tests/test_graph_components.py, within TestReasoningEngine class

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer with controlled outputs."""
        mock_model = mock.MagicMock()

        # Mock encode method to return a valid 1D embedding
        def mock_encode(
            sentences, convert_to_numpy=True, show_progress_bar=None, **kwargs
        ):
            # Added show_progress_bar parameter with default None
            # If single sentence, return 1D array
            if isinstance(sentences, str):
                return np.ones(384, dtype=np.float32) * 0.1

            # If multiple sentences, return 2D array
            return np.ones((len(sentences), 384), dtype=np.float32) * 0.1

        mock_model.encode = mock_encode
        return mock_model

    @pytest.fixture
    def embedding_engine(self, mock_sentence_transformer):
        """Fixture for an embedding engine with mock sentence transformer."""
        return EmbeddingEngine(dimension=384, sentence_model=mock_sentence_transformer)

    def test_initialization(self, embedding_engine, mock_sentence_transformer):
        """Test initialization of the embedding engine."""
        assert embedding_engine.index.d == 384
        assert embedding_engine.sentence_model is mock_sentence_transformer
        assert len(embedding_engine.nodes) == 0

    def test_add_node_embedding(self, embedding_engine):
        """Test adding node embeddings to the index."""
        # Add a node embedding
        embedding_engine.add_node_embedding("test_node", "Test content")

        # Check node was added
        assert len(embedding_engine.nodes) == 1
        assert embedding_engine.nodes[0] == "test_node"
        assert embedding_engine.index.ntotal == 1

    def test_find_similar_nodes_empty(self, embedding_engine):
        """Test finding similar nodes with empty index."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Search in empty index
            result = embedding_engine.find_similar_nodes("Test query")

            # Check result is empty
            assert result == []

    def test_find_similar_nodes(self, embedding_engine):
        """Test finding similar nodes."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Add several nodes
            embedding_engine.add_node_embedding("node1", "Machine Learning")
            embedding_engine.add_node_embedding("node2", "Artificial Intelligence")
            embedding_engine.add_node_embedding("node3", "Natural Language Processing")

            # Search for similar nodes
            result = embedding_engine.find_similar_nodes("AI and ML concepts")

            # Check result format
            assert isinstance(result, list)
            assert len(result) <= 3  # Should not exceed top_k

            # Since we're using a constant embedding in our mock,
            # the first nodes added should be returned first
            assert "node1" in result

    def test_compute_similarity(self, embedding_engine):
        """Test computing similarity between texts."""
        # Since we're using constant embeddings, similarity will be 1.0
        sim = embedding_engine.compute_similarity("Machine Learning", "AI technologies")

        # Check result - should be 1.0 with our mock
        assert 0 <= sim <= 1  # Similarity should be between 0 and 1
        assert sim == 1.0  # With our mock, it's exactly 1.0


class TestReasoningEngine:
    @pytest.fixture
    def graph_storage(self):
        """Fixture for a graph storage."""
        return GraphStorage()

    # In tests/test_graph_components.py, within TestReasoningEngine class

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer with controlled outputs."""
        mock_model = mock.MagicMock()

        # Mock encode method to return a valid 1D embedding
        def mock_encode(
            sentences, convert_to_numpy=True, show_progress_bar=None, **kwargs
        ):
            # Added show_progress_bar parameter with default None
            # If single sentence, return 1D array
            if isinstance(sentences, str):
                return np.ones(384, dtype=np.float32) * 0.1

            # If multiple sentences, return 2D array
            return np.ones((len(sentences), 384), dtype=np.float32) * 0.1

        mock_model.encode = mock_encode
        return mock_model

    @pytest.fixture
    def embedding_engine(self, mock_sentence_transformer):
        """Fixture for an embedding engine with mock sentence transformer."""
        return EmbeddingEngine(dimension=384, sentence_model=mock_sentence_transformer)

    @pytest.fixture
    def reasoning_engine(self, graph_storage, embedding_engine):
        """Fixture for a reasoning engine."""
        return ReasoningEngine(graph_storage, embedding_engine)

    def test_initialization(self, reasoning_engine, graph_storage, embedding_engine):
        """Test initialization of the reasoning engine."""
        assert reasoning_engine.graph is graph_storage
        assert reasoning_engine.embeddings is embedding_engine

    def test_update_from_chain_of_thought(self, reasoning_engine):
        """Test updating the graph from a chain of thought."""
        with mock.patch("graph_of_thoughts.graph_components.console.log"):
            # Create a chain of thought
            chain = ChainOfThought(
                nodes={"A": "Concept A", "B": "Concept B", "C": "Concept C"},
                edges=[["A", "B"], ["B", "C"]],
            )

            # Update graph from chain
            reasoning_engine.update_from_chain_of_thought(chain)

            # Check nodes were added with correct prefixes
            assert "reason_A" in reasoning_engine.graph.graph.nodes
            assert "reason_B" in reasoning_engine.graph.graph.nodes
            assert "reason_C" in reasoning_engine.graph.graph.nodes

            # Check edges were added with correct prefixes
            assert reasoning_engine.graph.graph.has_edge("reason_A", "reason_B")
            assert reasoning_engine.graph.graph.has_edge("reason_B", "reason_C")

            # Check node content
            assert reasoning_engine.graph.get_node_content("reason_A") == "Concept A"
            assert reasoning_engine.graph.get_node_content("reason_B") == "Concept B"
            assert reasoning_engine.graph.get_node_content("reason_C") == "Concept C"
