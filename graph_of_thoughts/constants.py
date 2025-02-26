# graph_of_thoughts/constants.py

from pathlib import Path

from rich.console import Console
from rich.theme import Theme

# TODO: Support a config / .env file for these settings and retrieve them using os.getenv or similar
# Custom theme for rich console
console = Console(
    theme=Theme(
        styles=dict(
            context="cyan",
            user="magenta",
            llm="green",
            prompt="blue",
            metrics="yellow",
            info="white",
            warning="bold red",
        )
    ),
)


MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 512


# Directory structure
MODULE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = Path((MODULE_DIR / "output").resolve())
FILES_DIR: Path = (MODULE_DIR / "files").resolve()

LLM_PATH: Path = Path(OUTPUT_DIR, "llm_graph.json")

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_CACHE_SIZE = 1000  # max number of embeddings to cache
# Thresholds
SIMILARITY_THRESHOLD = 0.75
IMPORTANCE_DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.8

# Prompts
SYSTEM_PROMPT = """
You are an AI assistant designed to generate a structured knowledge graph.

IMPORTANT RULES:

1️⃣ Output **ONLY** valid JSON inside <json>...</json> tags.
2️⃣ Structure must include:
    - "nodes": { "Concept": "Description" }
    - "edges": [ ["Parent", "Child"] ]
3️⃣ Expand the graph by:
   - **Adding new concepts** (subcategories, explanations, trade-offs).
   - **Connecting ideas** based on **causality**, **hierarchies**, and **dependencies**.
   - **Using technical depth** (e.g., caching → eviction policies → specific algorithms like LRU, LFU, ARC).

EXAMPLE OUTPUT:
<json>
{
    "nodes": {
        "Caching": "A method for storing frequently used data",
        "LRU": "Least Recently Used eviction strategy",
        "LFU": "Least Frequently Used strategy",
        "TTL": "A strategy based on expiration time",
        "ARC": "Adaptive Replacement Cache that balances LRU and LFU",
        "Write-Through": "Writes data to cache and database simultaneously",
        "Write-Back": "Writes data to cache first, then to the database"
    },
    "edges": [
        ["Caching", "LRU"],
        ["Caching", "LFU"],
        ["Caching", "TTL"],
        ["Caching", "ARC"],
        ["Caching", "Write-Through"],
        ["Caching", "Write-Back"],
        ["ARC", "LRU"],
        ["ARC", "LFU"]
    ]
}
</json>

📌 Your task: Expand the knowledge graph further based on the input query.
"""
