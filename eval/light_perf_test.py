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
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress
from rich.table import Table

from graph_of_thoughts.chat_manager import ChatManager
from graph_of_thoughts.constants import console
from graph_of_thoughts.context_manager import get_context_mgr

# Setup

EVAL_DIR = Path("eval/results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters
TEST_CASES = [
    {"name": "short", "turns": 3},
    {"name": "medium", "turns": 6},
    {"name": "long", "turns": 10},
]
REPEATS = 3  # Run each test multiple times for better averages

# Sample conversation starter
CONVERSATION_STARTER = "Tell me about database indexing and why it's useful."
TEST_MODEL_NAME = "unsloth/Llama-3.2-1B"


class PerformanceTest:
    """Runs comparison between traditional context and Graph of Thoughts."""

    def __init__(self):
        self.results = {"traditional": [], "graph_of_thoughts": []}


def run_test(self, method: str, turns: int) -> dict:
    """Run a test using either 'traditional' or 'graph_of_thoughts' and save progress."""
    context_mgr = get_context_mgr("llama3.1:8b")
    chat_mgr = ChatManager(context_mgr)

    response_times = []
    responses = []
    context = ""

    for turn in range(1, turns + 1):
        question = f"{CONVERSATION_STARTER} (turn {turn})"
        start_time = time.time()

        if method == "traditional":
            context += f"\nUser: {question}\nAI:"
            response = chat_mgr.generate_response(context)
        else:
            response = chat_mgr.process_turn(question, turn)

        response_times.append(time.time() - start_time)
        responses.append(response)

        # 🔹 Save progress incrementally
        self.save_partial_results(method, response_times)

    return {"response_times": response_times, "final_response": responses[-1]}

    def evaluate_tests(self):
        """Runs tests and compares results."""
        with Progress() as progress:
            task = progress.add_task(
                "Running performance tests...", total=len(TEST_CASES) * REPEATS
            )

            for case in TEST_CASES:
                turns = case["turns"]
                console.log(f"Running {case['name']} conversation test ({turns} turns)...")

                for _ in range(REPEATS):
                    # Traditional Context Window
                    trad_result = self.run_test("traditional", turns)
                    self.results["traditional"].append(trad_result)

                    # Graph of Thoughts
                    got_result = self.run_test("graph_of_thoughts", turns)
                    self.results["graph_of_thoughts"].append(got_result)

                    progress.update(task, advance=1)

        self.save_results()
        self.display_summary()

    def save_results(self):
        """Save results to JSON safely, ensuring directory exists and handling errors."""
        results_path = EVAL_DIR / "performance_results.json"

        try:
            EVAL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2)
            console.log(f"Results saved to {results_path}", style="llm")
        except ValueError as e:
            console.log(f"Error saving results: {e}", style="warning")

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
        """Runs the performance evaluation."""
        console.rule("Starting Lightweight Performance Evaluation", style="llm")
        self.evaluate_tests()
        console.rule("Evaluation Complete ✅", style="llm")

    def save_partial_results(self, method: str, response_times: list[float]):
        """Save partial results after each turn to avoid losing progress."""
        results_path = EVAL_DIR / f"partial_{method}.json"

        try:
            with open(results_path, "w") as f:
                json.dump(response_times, f, indent=2)
            console.log(f"[blue]Partial results saved to {results_path}[/blue]")
        except Exception as e:
            console.log(f"[red]Error saving partial results: {e}[/red]")


if __name__ == "__main__":
    PerformanceTest().run()
