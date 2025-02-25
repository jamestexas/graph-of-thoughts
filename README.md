# Graph of Thoughts

## Overview
**Graph of Thoughts** is an experimental approach to improving LLM (Large Language Model) context management by structuring memory as a dynamic, prunable knowledge graph instead of treating conversation history as a linear sequence of tokens.

This project explores how LLMs can maintain long-term coherence by storing their outputs and reasoning paths in a structured format, allowing for dynamic retrieval, prioritization, and decay of information.

## Core Idea
Instead of using traditional token-based conversation history, this system:
1. **Creates a Knowledge Graph** – User queries and LLM responses are stored as nodes.
2. **Embeds and Retrieves Context** – Nodes are embedded using sentence transformers for semantic search.
3. **Maintains Context Relevance** – Older or less relevant nodes decay over time and are pruned.
4. **Self-Documents Thought Processes** – The LLM outputs structured reasoning alongside responses.
5. **Serializes & Structures Memory** – The system attempts to enforce structured reasoning to improve recall and consistency.

## Repository Structure
```
├── README.md                 # This file
├── default_experiment_data.json  # Sample dataset for experiments
├── expirement/               # PoC scripts for testing ideas before committing them
│   ├── novel_supervisor_cot.py       # Experimenting with self-supervised CoT reasoning
│   └── poc_llm_self_graph.py         # Initial prototype of LLM-driven graph creation
├── graph_of_thoughts/        # Core implementation
│   ├── __init__.py
│   ├── chat_manager.py       # Handles LLM interactions and context injection
│   ├── constants.py          # Stores global constants
│   ├── context_manager.py    # Manages structured memory retrieval and decay
│   ├── debug_llm_output.txt  # Logs raw LLM outputs for debugging
│   ├── evaluate_llm_graph.py # Compares different graph structures
│   ├── files/                # Stores input queries and stuffed context examples
│   ├── generate_normal_outputs.py # Baseline LLM output generation (without graph enhancements)
│   ├── graph_components.py   # Core logic for knowledge graph creation
│   ├── main.py               # Entry point for running the system
│   ├── models.py             # Defines internal data structures for graph and memory
│   ├── output/               # Stores experiment results and generated graphs
│   ├── stress_test.py        # Tests system scalability under high load
│   ├── stuffed_context_test.py # Compares retrieval effectiveness
│   └── utils.py              # Helper functions
├── output/                   # Experiment results, including generated graphs
│   ├── baseline_graph.json   # Default reference graph
│   ├── llm_graph.json        # Graph generated from LLM outputs
├── pyproject.toml            # Python project configuration
└── uv.lock                   # Dependency lock file
```

## How It Works
1. **Graph Construction**: Each LLM interaction generates structured output, which is converted into a knowledge graph.
2. **Context Retrieval**: When a new query arrives, the system retrieves the most semantically relevant nodes instead of blindly stuffing past history into the prompt.
3. **Dynamic Memory Decay**: Older, less relevant nodes are pruned to keep context useful without exceeding token limits.
4. **Serialized Thought Process**: The LLM is encouraged to output structured JSON-like reasoning steps, making memory storage and retrieval more predictable.

## Goals & Challenges
### Goals:
- Efficiently manage long-term context in LLM interactions
- Improve coherence over multi-turn conversations
- Allow LLMs to track their own thought processes for better reasoning

### Challenges:
- **Structured Output Reliability** – LLMs are inconsistent in producing valid JSON for graph updates.
- **Efficient Pruning & Retrieval** – Ensuring important context remains while discarding unnecessary information.
- **Balancing Flexibility vs. Structure** – Avoiding over-constraining the LLM while keeping memory usable.

## Getting Started
### Installation
```bash
uv pip install -r requirements.txt
```

### Running the System
To start an experiment with structured memory:
```bash
python graph_of_thoughts/main.py
```

### Testing Different Approaches
- **Run baseline LLM inference (no graph)**:
  ```bash
  python graph_of_thoughts/generate_normal_outputs.py
  ```
- **Compare normal vs. graph-based context management**:
  ```bash
  python graph_of_thoughts/evaluate_llm_graph.py
  ```

## Next Steps
- Improve structured output validation (handling JSON failures, self-repair)
- Experiment with reinforcement learning to optimize memory decay functions
- Extend support for multi-modal (code snippets, images) knowledge graphs

## Contributing
This project is in active development. Feel free to submit issues, suggestions, or PRs to improve the structured memory and retrieval systems.

## License
MIT License

