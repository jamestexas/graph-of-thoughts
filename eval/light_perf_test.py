#!/usr/bin/env python
# light_performance_test.py

"""
Lightweight performance test for Graph of Thoughts vs. traditional context accumulation.
Measures:
- Response time
- Context recall
- Coherence across conversation turns
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress
from rich.table import Table

from graph_of_thoughts.chat_manager import ChatManager
from graph_of_thoughts.constants import console
from graph_of_thoughts.context_manager import get_context_mgr, parse_chain_of_thought
from graph_of_thoughts.utils import build_structured_prompt, build_llama_instruct_prompt

# Setup

EVAL_DIR = Path("eval/results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = EVAL_DIR / "traditional_baseline.json"
RESULTS_PATH = EVAL_DIR / "graph_of_thoughts_results.json"
PARTIAL_PATH = EVAL_DIR / "partial_results.json"

# Test parameters
TEST_CASES = [
    {"name": "short", "turns": 1},
    {"name": "medium", "turns": 2},
    {"name": "long", "turns": 3},
]
REPEATS = 3  # Run each test multiple times for better averages

# Sample conversation starter
CONVERSATION_STARTER = "Tell me about database indexing and why it's useful."
TEST_MODEL_NAME_3b = "unsloth/Llama-3.2-3B-Instruct"
TEST_MODEL_NAME = "unsloth/Llama-3.2-1B"


class PerformanceTest:
    """Runs comparison between traditional context and Graph of Thoughts."""

    def __init__(
        self,
        prompt_template: str,
        test_cases: list[str] = TEST_CASES,
        model_name: str = TEST_MODEL_NAME,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.test_cases = test_cases
        self.baseline = self._load_baseline()
        self.results = {
            "traditional": [],
            "graph_of_thoughts": [],
        }  # ✅ FIX: Ensure `self.results` exists
        self.context_manager = get_context_mgr(model_name=model_name)
        self.chat_manager = ChatManager(context_manager=self.context_manager)
        self.context = ""

    def _run_model(self, query: dict[str, Any]) -> str:
        """Runs the model using traditional context accumulation."""
        question = f"{CONVERSATION_STARTER} (turn {query['turns']})"
        self.context += f"\nUser: {question}\nAI:"

        structured_prompt = build_llama_instruct_prompt(
            system_text="",
            user_text=self.context,
        )

        # Traditional context accumulation
        response = self.chat_manager.generate_response(structured_prompt)
        return response

    def _run_graph_of_thoughts(self, query: dict[str, Any]) -> str:
        """Runs the Graph of Thoughts approach using ChatManager."""
        system_prompt = build_structured_prompt(query["name"])
        structured_prompt = build_llama_instruct_prompt(
            system_text=system_prompt,
            user_text=f"{CONVERSATION_STARTER} (turn {query['turns']})",
        )

        # Log the structured prompt for debugging
        console.log(f"[Structured Reasoning Prompt]: {structured_prompt}", style="prompt")

        # Graph of Thoughts handles context itself
        # Generate response using structured reasoning
        response = self.chat_manager.generate_response(structured_prompt)

        # Log the raw output
        console.log(f"[Raw LLM Output]: {response}", style="llm")

        try:
            chain_obj = parse_chain_of_thought(response)

            # Ensure output is valid JSON before returning
            return json.dumps({"nodes": chain_obj.nodes, "edges": chain_obj.edges}, indent=2)

        except Exception as e:
            console.log(f"[Error] JSON parsing failed: {e}", style="warning")
            return "{}"

    def _compare_responses(self, baseline_response: str, got_response: str):
        """Compares responses and logs mismatches."""
        if baseline_response != got_response:
            print(f"[WARNING] Response mismatch for query '{baseline_response[:50]}...'")

    def run_test(self, method: str, case: dict) -> dict:
        """Run a test using either 'traditional' or 'graph_of_thoughts' and save progress."""
        turns = case["turns"]
        query_name = case["name"]

        context_mgr = get_context_mgr(TEST_MODEL_NAME_3b)
        chat_mgr = ChatManager(context_mgr)

        response_times = []
        responses = []
        context = ""

        for turn in range(1, turns + 1):
            question = f"{query_name} (turn {turn})"
            start_time = time.time()

            if method == "traditional":
                context += f"\nUser: {question}\nAI:"
                result = chat_mgr.generate_response(context)
                structured = False
            else:
                result = chat_mgr.process_turn(question, turn)
                structured = True

            response_times.append(time.time() - start_time)
            responses.append(result)

            # Skip JSON parsing for Traditional responses
            if structured:
                try:
                    json.loads(result)  # Validate JSON
                except json.JSONDecodeError:
                    console.log(
                        f"[WARNING] GoT response missing valid JSON. Extracted:\n{result[:300]}...",
                        style="warning",
                    )

            # Save progress
            self.save_partial_results(method, response_times)

        return {
            "query": query_name,
            "response_times": response_times,
            "final_response": responses[-1],
        }

    def evaluate_tests(self):
        """Runs tests and compares results."""
        with Progress() as progress:
            task = progress.add_task(
                "Running performance tests...", total=len(self.test_cases) * REPEATS
            )

            for case in self.test_cases:  # Iterate through test cases
                turns = case["turns"]
                console.log(f"Running {case['name']} test ({turns} turns)...")

                for _ in range(REPEATS):
                    # Traditional Context Window
                    trad_result = self.run_test("traditional", turns)
                    self.results["traditional"].append(trad_result)

                    # Graph of Thoughts
                    got_result = self.run_test("graph_of_thoughts", turns)
                    self.results["graph_of_thoughts"].append(got_result)

                    progress.update(task, advance=1)

        self.save_results(self.results["graph_of_thoughts"])
        self.display_summary()

    def save_results(self, graph_of_thoughts_results: list[dict]):
        """
        Saves the results of the Graph of Thoughts evaluation.

        This function ensures that results are parsed correctly from JSON strings into Python dictionaries before
        writing them to a file. If no valid results exist after parsing, it will skip writing and log a warning.

        Args:
            graph_of_thoughts_results (list[dict]): A list of results, where each entry contains a query and its response.
        """

        # Convert responses from JSON strings to dictionaries before checking validity
        valid_results = []
        for entry in graph_of_thoughts_results:
            try:
                parsed_response = json.loads(entry["response"])  # Convert JSON string to dict
                entry["response"] = parsed_response  # Overwrite with structured dict
                valid_results.append(entry)
            except ValueError:
                console.log(
                    f"[ERROR] Failed to parse JSON response for query: {entry['query']}",
                    style="warning",
                )

        # Check if there are valid results to write
        if not valid_results:
            console.log(
                "[WARNING] No valid results to write. The output might be empty or malformed.",
                style="warning",
            )
            return  # Skip writing if all responses failed to parse

        # Save valid results to file
        results_path = "eval/results/graph_of_thoughts_results.json"
        try:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(valid_results, f, indent=2)
            console.log(f"[INFO] Successfully saved results to {results_path}!", style="bold green")
        except Exception as e:
            console.log(f"[ERROR] Failed to save results: {e}", style="warning")

    def display_summary(self):
        """Display test results in a table and generate a response time plot."""
        table = Table(title="Performance Test Results")
        table.add_column("Test", justify="left", style="cyan")
        table.add_column("Traditional Time (s)", justify="right", style="magenta")
        table.add_column("Graph of Thoughts Time (s)", justify="right", style="green")
        table.add_column("Improvement (%)", justify="right", style="yellow")

        avg_traditional = []
        avg_got = []

        for test_case, trad_data, got_data in zip(
            TEST_CASES, self.results["traditional"], self.results["graph_of_thoughts"]
        ):
            trad_avg = np.mean(trad_data["response_times"])
            got_avg = np.mean(got_data["response_times"])
            improvement = ((trad_avg - got_avg) / trad_avg) * 100 if trad_avg else 0

            avg_traditional.append(trad_avg)
            avg_got.append(got_avg)

            table.add_row(
                test_case["name"], f"{trad_avg:.3f}", f"{got_avg:.3f}", f"{improvement:+.1f}%"
            )

        console.print(table)

        # Plot Response Time Comparison
        plt.figure(figsize=(8, 5))
        labels = [case["name"] for case in TEST_CASES]
        plt.plot(labels, avg_traditional, label="Traditional Context", marker="o")
        plt.plot(labels, avg_got, label="Graph of Thoughts", marker="s")
        plt.xlabel("Conversation Length")
        plt.ylabel("Avg Response Time (s)")
        plt.title("Response Time Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(EVAL_DIR / "response_time_comparison.png")
        console.log(f"Graph saved to {EVAL_DIR / 'response_time_comparison.png'}", style="info")

    def run(self):
        console.rule("Starting Lightweight Performance Evaluation", style="llm")

        # Run traditional first, ensuring a baseline exists
        if not self.baseline:
            self.run_traditional()

        # Run Graph of Thoughts only if baseline is valid
        if self.baseline:
            got_results = self.run_graph_of_thoughts()
        else:
            print("[ERROR] Baseline missing. Skipping GoT evaluation.")

        self.save_results(got_results)
        console.rule("Evaluation Complete ✅", style="llm")
        self.display_summary()

    def save_partial_results(self, method: str, response_times: list[float]):
        """Save partial results after each turn to avoid losing progress."""
        results_path = EVAL_DIR / f"partial_{method}.json"

        try:
            # Load existing results if the file exists
            if results_path.exists():
                try:
                    text = results_path.read_text()
                    existing_data = json.loads(text)  # Fixed here
                except (json.JSONDecodeError, ValueError):
                    existing_data = []
            else:
                existing_data = []

            # Append new results
            existing_data.extend(response_times)

            # Save back to the file
            results_path.write_text(json.dumps(existing_data, indent=2))  # Fixed here

            console.log(f"Partial results updated in {results_path}")

        except Exception as e:
            console.log(f"Error saving partial results: {e}", style="warning")

    def _load_baseline(self) -> list[dict[str, Any]]:
        """Loads the baseline results if they exist and match model/prompt."""
        if BASELINE_PATH.exists():
            try:
                data = json.loads(BASELINE_PATH.read_text())
                if self._validate_baseline(data):
                    print("[INFO] Baseline loaded successfully.")
                    return data
                else:
                    print("[WARNING] Baseline model/prompt mismatch! Re-running Traditional Mode.")
            except (json.JSONDecodeError, ValueError):
                print("[WARNING] Baseline file is corrupted. Resetting baseline.")

        return []  # ✅ FIX: Ensure function always returns a valid list

    def _validate_baseline(self, data: list[dict[str, Any]]) -> bool:
        """Ensures stored baseline matches current model and prompt."""
        return all(
            entry["model"] == self.model_name and entry["prompt_template"] == self.prompt_template
            for entry in data
        )

    def run_traditional(self, force_rerun: bool = False):
        """Runs Traditional Mode once and stores results for validation."""
        if hasattr(self, "baseline") and self.baseline and not force_rerun:
            print("[INFO] Skipping Traditional Mode (baseline exists).")
            return self.baseline

        print("[INFO] Running Traditional Mode to establish baseline...")
        results = []
        for query in self.test_cases:
            start_time = time.time()
            response = self._run_model(query)
            elapsed = time.time() - start_time
            results.append({
                "query": query,
                "response": response,
                "model": self.model_name,
                "prompt_template": self.prompt_template,
                "response_time": elapsed,
            })

        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"[INFO] Traditional baseline saved to {BASELINE_PATH}")
        self.baseline = results
        return results

    def run_graph_of_thoughts(self):
        """Runs Graph of Thoughts and ensures a valid baseline exists."""
        if not hasattr(self, "baseline") or not self.baseline:
            raise ValueError("[ERROR] No baseline found! Run Traditional Mode first.")

        print("[INFO] Running Graph of Thoughts...")
        got_results = []
        for i, query in enumerate(self.test_cases):
            start_time = time.time()
            response = self._run_graph_of_thoughts(query)
            elapsed = time.time() - start_time

            baseline_response = self.baseline[i]["response"]
            self._compare_responses(baseline_response, response)

            got_results.append({"query": query, "response": response, "response_time": elapsed})

            # Save incremental results
            self.save_partial_results("graph_of_thoughts", got_results)

        RESULTS_PATH.write_text(json.dumps(got_results, indent=2))
        print(f"[INFO] Graph of Thoughts results saved to {RESULTS_PATH}")
        return got_results


if __name__ == "__main__":
    tester = PerformanceTest(prompt_template="default_prompt", test_cases=TEST_CASES)
    tester.run()
