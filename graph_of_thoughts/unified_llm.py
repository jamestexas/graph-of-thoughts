# graph_of_thoughts/unified_llm.py
# unified_llm.py
from typing import Any, Optional


class UnifiedLLM:
    """
    A unified interface that abstracts differences between HF transformer models
    and llama_cpp models.

    backend: "hf" for Hugging Face models, "llama_cpp" for llama_cpp models.
    model: The underlying model object.
    tokenizer: (Optional) The tokenizer (used only for HF backend).
    """

    def __init__(self, backend: str, model: Any, tokenizer: Optional[Any] = None):
        self.backend = backend.lower()
        self.model = model

        if self.backend == "hf":
            # Ensure the HF model is in evaluation mode.
            self.model.eval()
            self.tokenizer = self.model.tokenizer if tokenizer is None else tokenizer
        elif self.backend == "llama_cpp":
            self.tokenizer = self.model.tokenize if hasattr(self.model, "tokenize") else None
            # llama_cpp handles its own tokenization internally.
            pass
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompt: str, max_new_tokens: int = 200, **generate_kwargs) -> str:
        """
        Generate text from a prompt using the unified interface.
        """
        if self.backend == "hf":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # Pass additional generation kwargs if provided.
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif self.backend == "llama_cpp":
            # llama_cpp typically expects a 'prompt' and returns a dict with a "text" key.
            # Note: Adjust the keyword arguments as needed.
            response = self.model.generate(
                prompt=prompt, max_tokens=max_new_tokens, **generate_kwargs
            )
            return response.get("text", "")
        else:
            raise ValueError("Unsupported backend")

    def tokenize(self, text: str) -> Any:
        """
        Tokenize text. For HF models this uses the provided tokenizer; for llama_cpp,
        tokenization is handled internally (if available).
        """
        if self.backend == "hf":
            return self.tokenizer.tokenize(text)
        elif self.backend == "llama_cpp":
            if hasattr(self.model, "tokenize"):
                return self.model.tokenize(text)
            else:
                raise NotImplementedError("Tokenization not available for llama_cpp backend.")
        else:
            raise ValueError("Unsupported backend")
