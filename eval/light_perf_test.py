import json
import os
import time
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
import torch
from rich.table import Table
from transformers import GenerationConfig

from graph_of_thoughts.constants import MODEL_NAME, console
from graph_of_thoughts.context_manager import ContextGraphManager, get_context_mgr
from graph_of_thoughts.models import ChainOfThought
from graph_of_thoughts.utils import get_llm_model, get_tokenizer


class TestCase(TypedDict):
    name: str
    turns: list[str]


class LightPerformanceTest:
    """Lightweight performance testing for Graph of Thoughts."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = get_llm_model(model_name=model_name)
        self.tokenizer = get_tokenizer(model_name=model_name)
        short_test = TestCase(
            name="short",
            turns=["Tell me about database indexing and why it's useful."],
        )
        medium_test = TestCase(
            name="medium",
            turns=[
                "Tell me about database indexing and why it's useful.",
                "What are the different types of database indices?",
            ],
        )
        long_test = TestCase(
            name="long",
            turns=[
                "Tell me about database indexing and why it's useful.",
                "What are the different types of database indices?",
                "When would you choose a B-tree versus a hash index?",
            ],
        )
        # Test complexity levels
        self.tests = dict(
            short=short_test,
            medium=medium_test,
            long=long_test,
        )

        # Setup directories
        self.results_dir = Path("eval/results")
        self.cache_dir = Path("eval/cache")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Results storage
        self.results = {}

    def run_llm_inference(self, prompt):
        """Run inference with the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=800,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result

    def get_context_manager(self):
        """Get a fresh context manager."""
        return get_context_mgr(model_name=self.model_name)

    def build_traditional_prompt(self, query_data):
        """Build a traditional prompt without graph structure."""
        system_prompt = """
        You are an AI assistant designed to answer questions accurately and helpfully.
        Please respond to the following query with a detailed and informative answer.
        """

        # For multi-turn conversations, build a conversation history
        if len(query_data["turns"]) > 1:
            conversation = []
            for i, turn in enumerate(query_data["turns"]):
                conversation.append(f"User Turn {i + 1}: {turn}")

            prompt = f"{system_prompt}\n\nConversation History:\n{' '.join(conversation)}\n\nPlease respond to the latest query."
        else:
            prompt = f"{system_prompt}\n\nQuery: {query_data['turns'][0]}"

        return prompt

    def build_structured_prompt(self, query, context=""):
        """Build a structured reasoning prompt for graph of thoughts."""
        system_prompt = """
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
        """

        # Include context if available
        context_section = f"\nRelevant Context:\n{context}\n" if context else ""

        prompt = f"{system_prompt}{context_section}\nQuestion: {query}\n<json>"
        return prompt

    def extract_chain_of_thought(self, response):
        """Extract a chain of thought from the response."""
        import re

        # Extract JSON content
        json_pattern = re.compile(r"<json>(.*?)</json>", re.DOTALL)
        match = json_pattern.search(response)

        if not match:
            raise ValueError("No JSON found in response")

        json_str = match.group(1).strip()
        data = json.loads(json_str)

        # Validate required fields
        if "nodes" not in data or "edges" not in data:
            raise ValueError("Missing required fields in JSON")

        return ChainOfThought(nodes=data["nodes"], edges=data["edges"])

    def get_traditional_response(self, query_data):
        """Get traditional LLM response."""
        cache_file = self.cache_dir / f"traditional_{query_data['name']}.json"
        query_hash = f"Traditional response to '{str(query_data)[:50]}...'"

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_responses = json.load(f)

                if query_hash in cached_responses:
                    print(f"[INFO] Found cached response for {query_hash}")
                    return cached_responses[query_hash]
            except Exception as e:
                print(f"[WARNING] Error loading cache: {e}")
                cached_responses = {}
        else:
            cached_responses = {}

        # Generate new response
        print(f"[INFO] Generating new response for {query_hash}")
        prompt = self.build_traditional_prompt(query_data)
        start_time = time.time()
        response = self.run_llm_inference(prompt)
        elapsed = time.time() - start_time

        # Store in cache
        result = {"response": response, "time": elapsed}
        cached_responses[query_hash] = result

        # Save cache
        try:
            with open(cache_file, "w") as f:
                json.dump(cached_responses, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Error saving cache: {e}")

        return result

    def run_graph_of_thoughts(self, query_data):
        """Run Graph of Thoughts approach."""
        context_manager = self.get_context_manager()

        start_time = time.time()
        results = []

        # Process each turn
        for turn_idx, turn in enumerate(query_data["turns"]):
            print(
                f"[INFO] Processing turn {turn_idx + 1}/{len(query_data['turns'])}: {turn[:50]}..."
            )

            # Add to context
            context_manager.add_context(f"user_turn_{turn_idx}", turn)

            # Get relevant context
            relevant_context = context_manager.query_context(turn, top_k=3)
            context_text = "\n".join([
                context_manager.graph_storage.get_node_content(ctx) or ""
                for ctx in relevant_context
                if ctx in context_manager.graph_storage.graph
            ])

            # Build prompt
            structured_prompt = self.build_structured_prompt(turn, context=context_text)

            # Generate response
            response = self.run_llm_inference(structured_prompt)

            # Add response to context
            context_manager.add_context(f"llm_turn_{turn_idx}", response)

            # Try to extract structured reasoning
            try:
                chain_obj = self.extract_chain_of_thought(response)

                # Add nodes and edges to context
                for node_id, content in chain_obj.nodes.items():
                    context_manager.add_context(f"reason_{node_id}_{turn_idx}", content)

                # Add edges
                for source, target in chain_obj.edges:
                    source_id = f"reason_{source}_{turn_idx}"
                    target_id = f"reason_{target}_{turn_idx}"
                    if (
                        source_id in context_manager.graph_storage.graph
                        and target_id in context_manager.graph_storage.graph
                    ):
                        context_manager.graph_storage.add_edge(source_id, target_id)
            except Exception as e:
                print(f"[WARNING] Failed to extract chain of thought: {e}")

            # Decay and prune
            for node_id in list(context_manager.graph_storage.graph.nodes()):
                context_manager.graph_storage.decay_node_importance(node_id)
            context_manager.prune_context(threshold=0.7)

            # Store result
            results.append({"turn": turn, "response": response})

        elapsed = time.time() - start_time
        return {"results": results, "time": elapsed}

    def run_test(self, test_name):
        """Run test for a specific complexity level."""
        test_data = self.tests[test_name]
        print(f"\n[INFO] Running test: {test_name} ({len(test_data['turns'])} turns)")

        # Run traditional approach
        print("\n[INFO] Running traditional approach...")
        trad_result = self.get_traditional_response(test_data)

        # Run Graph of Thoughts
        print("\n[INFO] Running Graph of Thoughts...")
        got_result = self.run_graph_of_thoughts(test_data)

        # Store results
        self.results[test_name] = {
            "traditional_times": [trad_result["time"]],
            "got_times": [got_result["time"]],
            "traditional_responses": [trad_result["response"]],
            "got_responses": got_result["results"],
        }

        # Print results
        print(f"\n[INFO] Test {test_name} completed:")
        print(f"  - Traditional time: {trad_result['time']:.2f}s")
        print(f"  - Graph of Thoughts time: {got_result['time']:.2f}s")
        if trad_result["time"] > 0:
            improvement = (
                (trad_result["time"] - got_result["time"]) / trad_result["time"] * 100
            )
            print(f"  - Improvement: {improvement:.2f}%")

    def display_summary(self):
        """Display performance summary and charts."""
        # Check if we have results
        if not self.results:
            print("[WARNING] No results to display.")
            return

        # Calculate averages
        avg_traditional = []
        avg_got = []
        improvement = []
        labels = []

        for test_name, test_results in self.results.items():
            labels.append(test_name)

            # Traditional times
            if test_results.get("traditional_times"):
                trad_avg = sum(test_results["traditional_times"]) / len(
                    test_results["traditional_times"]
                )
                avg_traditional.append(trad_avg)
            else:
                avg_traditional.append(0)

            # GoT times
            if test_results.get("got_times"):
                got_avg = sum(test_results["got_times"]) / len(
                    test_results["got_times"]
                )
                avg_got.append(got_avg)
            else:
                avg_got.append(0)

            # Calculate improvement
            if avg_traditional[-1] > 0:
                imp = (avg_traditional[-1] - avg_got[-1]) / avg_traditional[-1] * 100
                improvement.append(imp)
            else:
                improvement.append(0)

        # Display table
        table = Table(title="Performance Test Results")
        table.add_column("Test")
        table.add_column("Traditional Time (s)")
        table.add_column("Graph of Thoughts Time (s)")
        table.add_column("Improvement (%)")

        for i, test in enumerate(labels):
            table.add_row(
                test,
                f"{avg_traditional[i]:.2f}",
                f"{avg_got[i]:.2f}",
                f"{improvement[i]:.2f}",
            )
        divisor = (console.width * 0.5 if console.width else 80) * "-"
        console.print(
            f"{divisor} Evaluation Complete ✅ {divisor}\n",
            style="bold green",
        )
        console.print(table)

        # Create plot if we have matplotlib
        try:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(labels))
            width = 0.35

            # Ensure arrays are not empty
            if avg_traditional and avg_got:
                plt.bar(
                    x - width / 2, avg_traditional, width, label="Traditional Context"
                )
                plt.bar(x + width / 2, avg_got, width, label="Graph of Thoughts")

                plt.xlabel("Test Complexity")
                plt.ylabel("Average Time (seconds)")
                plt.title("Performance Comparison")
                plt.xticks(x, labels)
                plt.legend()

                # Save plot
                plot_path = os.path.join(self.results_dir, "performance_comparison.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"[INFO] Plot saved to {plot_path}")
        except Exception as e:
            print(f"[WARNING] Error generating plot: {e}")

    def save_results(self):
        """Save results to file."""
        if not self.results:
            print("[WARNING] No results to save.")
            return

        # Save Graph of Thoughts results
        got_file = os.path.join(self.results_dir, "graph_of_thoughts_results.json")
        with open(got_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[INFO] Graph of Thoughts results saved to {got_file}")

        try:
            # Create a summary file
            summary_file = os.path.join(self.results_dir, "performance_summary.json")
            summary = {}

            for test_name, test_results in self.results.items():
                trad_time = (
                    sum(test_results["traditional_times"])
                    / len(test_results["traditional_times"])
                    if test_results.get("traditional_times")
                    else 0
                )
                got_time = (
                    sum(test_results["got_times"]) / len(test_results["got_times"])
                    if test_results.get("got_times")
                    else 0
                )

                improvement = 0
                if trad_time > 0:
                    improvement = (trad_time - got_time) / trad_time * 100

                summary[test_name] = {
                    "traditional_time": trad_time,
                    "got_time": got_time,
                    "improvement": improvement,
                }

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Successfully saved results to {summary_file}!")
        except Exception as e:
            print(f"[WARNING] Error saving summary: {e}")

    def run(self):
        """Run all tests."""
        # Run tests for each complexity level
        for test_name in self.tests.keys():
            self.run_test(test_name)

        # Save results
        self.save_results()

        # Display summary
        self.display_summary()


if __name__ == "__main__":
    tester = LightPerformanceTest()
    tester.run()
