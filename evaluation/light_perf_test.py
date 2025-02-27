import json
import os
import re
import time
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from rich.table import Table
from run_gbnf_conversation import MODEL_PATH as LLAMA_MODEL_PATH, get_grammar_path
from transformers import GenerationConfig

from graph_of_thoughts.constants import console
from graph_of_thoughts.context_manager import ContextGraphManager, DateTimeEncoder, get_context_mgr
from graph_of_thoughts.models import ChainOfThought
from graph_of_thoughts.utils import get_unified_llm_model


class PerformanceTestCase(TypedDict):
    """Represents a performance test case."""

    name: str
    turns: list[str]


SHORT_CASE = PerformanceTestCase(
    name="short",
    turns=["Tell me about database indexing and why it's useful."],
)
MED_CASE = PerformanceTestCase(
    name="medium",
    turns=[
        "Tell me about database indexing and why it's useful.",
        "What are the different types of database indices?",
    ],
)
LONG_TEST = PerformanceTestCase(
    name="long",
    turns=[
        "Tell me about database indexing and why it's useful.",
        "What are the different types of database indices?",
        "When would you choose a B-tree versus a hash index?",
    ],
)


# Non-technical conversation test cases
NON_TECH_SHORT = PerformanceTestCase(
    name="nontech_short",
    turns=["What's the meaning of life?"],
)
NON_TECH_MED = PerformanceTestCase(
    name="nontech_medium",
    turns=["What's the meaning of life?", "How do different cultures interpret it?"],
)
NON_TECH_LONG = PerformanceTestCase(
    name="nontech_long",
    turns=[
        "What's the meaning of life?",
        "How do different cultures interpret it?",
        "Can you give an example of how this shapes societal values?",
    ],
)

# Emotional conversation test cases
EMOTIONAL_SHORT = PerformanceTestCase(
    name="emotional_short",
    turns=["I'm feeling really overwhelmed. Can you help me understand what's wrong?"],
)
EMOTIONAL_MED = PerformanceTestCase(
    name="emotional_medium",
    turns=[
        "I'm feeling really overwhelmed. Can you help me understand what's wrong?",
        "Sometimes, I feel like I'm not enough and I'm struggling with self-doubt.",
    ],
)
EMOTIONAL_LONG = PerformanceTestCase(
    name="emotional_long",
    turns=[
        "I'm feeling really overwhelmed. Can you help me understand what's wrong?",
        "Sometimes, I feel like I'm not enough and I'm struggling with self-doubt.",
        "What strategies can I use to build my confidence and manage these emotions?",
    ],
)

# Creative conversation test cases
CREATIVE_SHORT = PerformanceTestCase(
    name="creative_short",
    turns=["Write a short poem about autumn."],
)
CREATIVE_MED = PerformanceTestCase(
    name="creative_medium",
    turns=[
        "Write a short poem about autumn.",
        "Now, explain how the poem reflects the changing seasons and the passage of time.",
    ],
)
CREATIVE_LONG = PerformanceTestCase(
    name="creative_long",
    turns=[
        "Write a short poem about autumn.",
        "Now, explain how the poem reflects the changing seasons and the passage of time.",
        "What emotions does autumn evoke, and how can these relate to personal growth?",
    ],
)

# Construct a comprehensive dictionary
PERFORMANCE_TEST_CASES = dict(
    short=SHORT_CASE,
    medium=MED_CASE,
    long=LONG_TEST,
    nontech_short=NON_TECH_SHORT,
    nontech_medium=NON_TECH_MED,
    nontech_long=NON_TECH_LONG,
    emotional_short=EMOTIONAL_SHORT,
    emotional_medium=EMOTIONAL_MED,
    emotional_long=EMOTIONAL_LONG,
    creative_short=CREATIVE_SHORT,
    creative_medium=CREATIVE_MED,
    creative_long=CREATIVE_LONG,
)
TEST_MODEL_NAME = "unsloth/Llama-3.2-1B"
TEST_MODEL_3B = "unsloth/Llama-3.2-3B-Instruct"


class LightPerformanceTest:
    """Lightweight performance testing for Graph of Thoughts."""

    def __init__(
        self,
        model_name=TEST_MODEL_NAME,
        use_llama: bool = True,
    ) -> None:
        """
        Initialize the performance test with a specific model.
        """
        if use_llama:
            # Use model_path instead of model_name for llama_cpp
            self.model_name = str(LLAMA_MODEL_PATH)
            kwargs = dict(
                backend="llama_cpp",
                model_path=self.model_name,  # change from model_name to model_path
                grammar_file=str(get_grammar_path()),
            )

        else:
            kwargs = dict(
                backend="hf",
                model_name=model_name,
            )
            self.model_name = model_name

        self.model = get_unified_llm_model(**kwargs)
        self.tokenizer = self.model.tokenizer

        # Create the context manager with the unified model
        self.context_manager = ContextGraphManager(
            tokenizer=self.tokenizer,
            model=self.model,
        )
        # Test complexity levels
        self.tests = PERFORMANCE_TEST_CASES.copy()

        # Setup directories
        self.results_dir = Path("evaluation/results")
        self.cache_dir = Path("evaluation/cache")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Results storage
        self.results = {}

    def run_llm_inference(self, prompt: str) -> str:
        """Run inference with the LLM."""
        max_new_tokens = 800  # or your desired value
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
        # For llama_cpp backend, our unified generate() already returns a string.
        if self.model.backend == "llama_cpp":
            return output
        else:
            # For HF backend, output is a tensor list that needs decoding.
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

IMPORTANT INSTRUCTIONS:
1. You must respond with a valid JSON structure containing nodes and edges.
2. Your response format must follow this exact pattern:

<json>
{
"nodes": {
"concept1": "description of concept 1",
"concept2": "description of concept 2",
"concept3": "description of concept 3"
},
"edges": [
["concept1", "concept2"],
["concept1", "concept3"]
]
}
</json>

3. Ensure your JSON includes:
- Multiple concepts related to the question
- Meaningful connections between concepts
- Clear and descriptive explanations
4. Do not include any text outside the <json> tags
5. Make sure your JSON is properly formatted with quotes around keys and string values
"""

        # Include context if available
        context_section = f"\nRelevant Context:\n{context}\n" if context else ""

        prompt = f"{system_prompt}{context_section}\nQuestion: {query}\n\nRemember to respond with valid JSON in <json> tags."
        return prompt

    def extract_chain_of_thought(self, response):
        """Extract a chain of thought from the response with improved error handling."""

        # Log the raw response for debugging
        console.log("[Debug] Raw Response (first 500 chars):", style="dim")
        console.log(response[:500] + ("..." if len(response) > 500 else ""), style="dim")

        # Extract JSON content
        json_pattern = re.compile(r"<json>(.*?)</json>", re.DOTALL)
        match = json_pattern.search(response)

        if not match:
            console.log("[Error] No JSON tags found in response", style="warning")
            # Try to find any JSON-like structure
            fallback_pattern = re.compile(r"\{[\s\S]*?\}", re.DOTALL)
            fallback_match = fallback_pattern.search(response)

            if fallback_match:
                json_str = fallback_match.group(0).strip()
                console.log("[Debug] Found fallback JSON-like structure", style="dim")
            else:
                console.log("[Error] No JSON-like structure found", style="error")
                # Return empty chain of thought
                return ChainOfThought(nodes={}, edges=[])
        else:
            json_str = match.group(1).strip()

        # Parse the JSON with error handling
        try:
            data = json.loads(json_str)

            # Validate required fields
            if "nodes" not in data or "edges" not in data:
                console.log("[Error] Missing required fields in JSON", style="warning")
                console.log(f"DATA FOUND: {data}", style="bold yellow")
                # Initialize missing fields
                data = {"nodes": data.get("nodes", {}), "edges": data.get("edges", [])}

            return ChainOfThought(nodes=data["nodes"], edges=data["edges"])

        except json.JSONDecodeError as e:
            console.log(f"[Error] JSON decode error: {e}", style="warning")
            console.log(f"[Debug] Problematic JSON string: {json_str[:200]}", style="dim")
            # Return empty chain of thought
            return ChainOfThought(nodes={}, edges=[])

    def get_traditional_response(self, query_data):
        """Get traditional LLM response."""
        cache_file: Path = self.cache_dir / f"traditional_{query_data['name']}.json"
        query_hash = f"Traditional response to '{str(query_data)[:50]}...'"

        # Try to load from cache
        if cache_file.exists():
            try:
                contents = cache_file.read_text()
                cached_responses = json.loads(contents)

                if query_hash in cached_responses:
                    console.log(f"[INFO] Found cached response for {query_hash} in {cache_file}")
                    return cached_responses[query_hash]
            except Exception as e:
                console.log(f"[WARNING] Error loading cache: {e}")
                cached_responses = {}
        else:
            cached_responses = {}

        # Generate new response
        console.log(f"[INFO] Generating new response for {query_hash}")
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
            console.log(f"[WARNING] Error saving cache: {e}")

        return result

    def run_graph_of_thoughts(self, query_data):
        """Run the Graph of Thoughts approach using the chat_manager pattern."""
        # Initialize chat manager with a fresh context manager
        from graph_of_thoughts.chat_manager import ChatManager

        context_manager = self.get_context_manager()
        chat_manager = ChatManager(context_manager)

        start_time = time.time()
        results = []

        # Process each turn in the conversation
        for turn_idx, turn in enumerate(query_data["turns"]):
            console.log(
                f"[INFO] Processing turn {turn_idx + 1}/{len(query_data['turns'])}: {turn[:50]}..."
            )

            # Use the existing process_turn method
            response = chat_manager.process_turn(turn, turn_idx + 1)

            # Store the result
            results.append({"turn": turn, "response": response})

        elapsed = time.time() - start_time

        # Save the final graph state for visualization
        graph_file = os.path.join(self.results_dir, f"graph_{query_data['name']}.json")

        # Use the improved save_graph_state method
        if hasattr(context_manager, "save_graph_state"):
            success = context_manager.save_graph_state(graph_file)
            if not success:
                # Fallback to manual saving
                try:
                    graph_data = context_manager.graph_to_json()
                    with open(graph_file, "w") as f:
                        json.dump(graph_data, f, cls=DateTimeEncoder, indent=2)
                    console.log(f"[INFO] Graph state saved to {graph_file} (fallback method)")
                except Exception as e:
                    console.log(f"[WARNING] Error saving graph state: {e}")
        else:
            # Use the original approach with improved encoder
            try:
                graph_data = context_manager.graph_to_json()
                with open(graph_file, "w") as f:
                    json.dump(graph_data, f, cls=DateTimeEncoder, indent=2)
                console.log(f"[INFO] Graph state saved to {graph_file}")
            except Exception as e:
                console.log(f"[WARNING] Error saving graph state: {e}")

        return {"results": results, "time": elapsed}

    def run_test(self, test_name):
        """Run test for a specific complexity level."""
        test_data = self.tests[test_name]
        console.log(f"\n[INFO] Running test: {test_name} ({len(test_data['turns'])} turns)")

        # Run traditional approach
        console.log("\n[INFO] Running traditional approach...")
        trad_result = self.get_traditional_response(test_data)

        # Run Graph of Thoughts
        console.log("\n[INFO] Running Graph of Thoughts...")
        got_result = self.run_graph_of_thoughts(test_data)

        # Store results
        self.results[test_name] = {
            "traditional_times": [trad_result["time"]],
            "got_times": [got_result["time"]],
            "traditional_responses": [trad_result["response"]],
            "got_responses": got_result["results"],
        }

        # Print results
        console.log(f"\n[INFO] Test {test_name} completed:")
        console.log(f"  - Traditional time: {trad_result['time']:.2f}s")
        console.log(f"  - Graph of Thoughts time: {got_result['time']:.2f}s")
        if trad_result["time"] > 0:
            improvement = (trad_result["time"] - got_result["time"]) / trad_result["time"] * 100
            console.log(f"  - Improvement: {improvement:.2f}%")

    def display_summary(self):
        """Display performance summary and charts."""
        # Check if we have results
        if not self.results:
            console.log("[WARNING] No results to display.")
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
                got_avg = sum(test_results["got_times"]) / len(test_results["got_times"])
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
        divisor_length = int(console.width * 0.5) if console.width else 80
        divisor = "-" * divisor_length
        console.log(
            f"{divisor} Evaluation Complete ✅ {divisor}",
            style="bold green",
        )
        console.log(table)

        # Create plot if we have matplotlib
        try:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(labels))
            width = 0.35

            # Ensure arrays are not empty
            if avg_traditional and avg_got:
                plt.bar(x - width / 2, avg_traditional, width, label="Traditional Context")
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
                console.log(f"[INFO] Plot saved to {plot_path}")
        except Exception as e:
            console.log(f"[WARNING] Error generating plot: {e}")

    def save_results(self):
        """Save results to file."""
        if not self.results:
            console.log("[WARNING] No results to save.")
            return

        # Save Graph of Thoughts results
        got_file = os.path.join(self.results_dir, "graph_of_thoughts_results.json")
        with open(got_file, "w") as f:
            json.dump(self.results, f, indent=2)
        console.log(f"[INFO] Graph of Thoughts results saved to {got_file}")

        try:
            # Create a summary file
            summary_file = os.path.join(self.results_dir, "performance_summary.json")
            summary = {}

            for test_name, test_results in self.results.items():
                trad_time = (
                    sum(test_results["traditional_times"]) / len(test_results["traditional_times"])
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
            console.log(f"[INFO] Successfully saved results to {summary_file}!")
        except Exception as e:
            console.log(f"[WARNING] Error saving summary: {e}")

    def run(self):
        """Run all tests."""
        categories = {}
        for test_name, test_data in self.tests.items():
            try:
                console.log(f"Running test: {test_name}", style="bold  yellow")
                start = time.time()
                self.run_test(test_name)
                elapsed = time.time() - start
                categories[test_name] = elapsed
            except Exception as e:
                console.log(f"Test {test_name} failed: {e}", style="bold red")

        self.save_results()
        self.display_summary()
        console.log(f"Performance tests completed in {categories}")


if __name__ == "__main__":
    tester = LightPerformanceTest()
    tester.run()
