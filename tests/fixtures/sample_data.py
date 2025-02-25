# tests/fixtures/sample_data.py

from graph_of_thoughts.models import SeedData

# Sample seed data for testing
SAMPLE_SEED_DATA = [
    SeedData(
        title="Machine Learning",
        content="A field of study that gives computers the ability to learn without being explicitly programmed.",
    ),
    SeedData(
        title="Neural Networks",
        content="Computing systems inspired by the biological neural networks that constitute animal brains.",
    ),
    SeedData(
        title="Reinforcement Learning",
        content="An area of machine learning concerned with how software agents ought to take actions in an environment to maximize some notion of cumulative reward.",
    ),
]

# Sample user inputs for testing
SAMPLE_USER_INPUTS = [
    "Tell me about neural networks and their applications",
    "How does reinforcement learning work?",
    "What are the differences between supervised and unsupervised learning?",
]

# Sample LLM responses for mocking
SAMPLE_LLM_RESPONSES = [
    """
<json>
{
    "nodes": {
        "Neural Networks": "Computing systems inspired by biological neural networks",
        "Applications": "Various practical uses of neural networks",
        "Image Recognition": "Using neural networks to identify objects in images",
        "Natural Language Processing": "Using neural networks to understand and generate human language"
    },
    "edges": [
        ["Neural Networks", "Applications"],
        ["Applications", "Image Recognition"],
        ["Applications", "Natural Language Processing"]
    ]
}
</json>
    """,
    """
<json>
{
    "nodes": {
        "Reinforcement Learning": "Learning through interaction with an environment",
        "Agent": "The decision-making entity in reinforcement learning",
        "Environment": "The external system with which the agent interacts",
        "Reward Signal": "Feedback that guides the agent's learning",
        "Policy": "The strategy used by the agent to decide which actions to take"
    },
    "edges": [
        ["Reinforcement Learning", "Agent"],
        ["Reinforcement Learning", "Environment"],
        ["Reinforcement Learning", "Reward Signal"],
        ["Agent", "Policy"],
        ["Environment", "Reward Signal"],
        ["Reward Signal", "Agent"]
    ]
}
</json>
    """,
    """
<json>
{
    "nodes": {
        "Machine Learning": "A field of study that gives computers the ability to learn",
        "Supervised Learning": "Learning with labeled training data",
        "Unsupervised Learning": "Learning from unlabeled data",
        "Training Data": "Data used to train machine learning models",
        "Labels": "Target outputs for supervised learning",
        "Clustering": "Grouping similar data points in unsupervised learning"
    },
    "edges": [
        ["Machine Learning", "Supervised Learning"],
        ["Machine Learning", "Unsupervised Learning"],
        ["Supervised Learning", "Training Data"],
        ["Supervised Learning", "Labels"],
        ["Unsupervised Learning", "Clustering"],
        ["Training Data", "Labels"]
    ]
}
</json>
    """,
]
