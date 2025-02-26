# Graph of Thoughts: Knowledge Graph Memory for LLMs

<p align="center">
  <img src="docs/graph_illustration.png" alt="Graph of Thoughts Illustration" width="600"/>
</p>

## 🧠 The Challenge of LLM Memory

Large Language Models typically handle context through a linear sequence of tokens, leading to several limitations:

- **Limited Memory Span**: Context windows have a fixed size, causing distant information to be forgotten
- **No Prioritization**: All context receives equal importance regardless of relevance
- **Linear Structure**: Relationships between concepts aren't explicitly captured
- **Inconsistency**: LLMs can lose track of their own reasoning across long conversations

## 💡 Our Approach: Structured Memory as a Graph

**Graph of Thoughts** reimagines LLM memory as a dynamic, structured knowledge graph that enables:

1. **Semantic Retrieval**: Fetch only the most relevant context based on semantic similarity
2. **Priority-Based Decay**: More important or recent information persists longer
3. **Relationship Preservation**: Explicitly capture connections between concepts
4. **Reasoning Traceability**: Follow the model's thought process through graph exploration

## 🔍 Core Concepts

### 1. Memory as Graph Structure
Instead of a token window, we store information in a directed graph where:
- **Nodes** represent concepts, facts, or user inputs
- **Edges** capture relationships and dependencies between nodes
- **Embeddings** enable semantic similarity search
- **Importance scores** determine which nodes to keep or prune

### 2. LLM-Generated Structure
The LLM itself participates in creating its memory structure by:
- Generating structured JSON representing its reasoning process
- Identifying key concepts and relationships
- Updating the graph with new knowledge
- Following chains of thought through the graph

### 3. Dynamic Context Management
The system maintains context relevance by:
- Retrieving the most semantically similar nodes for each query
- Automatically decaying node importance over time
- Pruning less relevant information when the graph grows too large
- Preserving critical paths of reasoning even as details fade

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt