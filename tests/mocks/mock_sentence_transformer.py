# tests/mocks/mock_sentence_transformer.py

"""
Mock implementation of SentenceTransformer for testing.
This avoids loading the actual model during tests, which can be slow and resource-intensive.
"""

import numpy as np


class MockSentenceTransformer:
    """
    Mock implementation of SentenceTransformer that returns consistent embeddings
    based on the input text for testing purposes.
    """

    def __init__(self, model_name_or_path=None):
        """Initialize the mock transformer."""
        self.model_name = model_name_or_path or "mock-model"
        self.device = "cpu"
        self.embedding_dim = 384  # Standard dimension for sentence-transformers

        # Dictionary to store predefined semantic relationships
        self.semantic_pairs = {
            # Format: (text1, text2): similarity_score
            ("Machine Learning", "AI"): 0.85,
            ("Neural Networks", "Deep Learning"): 0.9,
            ("Supervised Learning", "Training with Labels"): 0.88,
            ("A → B", "A → B"): 1.0,
            ("A → B", "X → Y"): 0.8,  # Simulating a semantically similar edge
            ("B → C", "B → D"): 0.5,  # Simulating a moderately similar edge
        }

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
        device=None,
        normalize_embeddings=False,
    ):
        """
        Mock implementation of the encode method.
        Returns deterministic embeddings based on the input text.
        NOTE: Returns numpy arrays (not lists) to support matrix operations like @
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        result = []
        for sentence in sentences:
            # Create a deterministic embedding based on the hash of the sentence
            # This ensures the same sentence always gets the same embedding
            hash_val = hash(sentence) % 10000
            np.random.seed(hash_val)

            # Generate a pseudo-random embedding
            embedding = np.random.normal(0, 1, self.embedding_dim)

            # Normalize the embedding to unit length
            embedding = embedding / np.linalg.norm(embedding)

            result.append(embedding)

        # Important: Return as numpy array, not a list!
        return np.array(result)

    def compute_similarity(self, text1, text2):
        """
        Compute semantic similarity between two texts.
        Uses predefined values for certain pairs, or calculates based on embeddings.
        """
        # Check if we have a predefined similarity for this pair
        if (text1, text2) in self.semantic_pairs:
            return self.semantic_pairs[(text1, text2)]
        elif (text2, text1) in self.semantic_pairs:
            return self.semantic_pairs[(text2, text1)]

        # Otherwise, calculate similarity from embeddings
        embeddings = self.encode([text1, text2])
        return float(np.dot(embeddings[0], embeddings[1]))


def get_mock_sentence_transformer(model_name=None):
    """Factory function to create a mock sentence transformer."""
    return MockSentenceTransformer(model_name)
