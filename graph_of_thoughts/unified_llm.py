# graph_of_thoughts/unified_llm.py


from typing import Any


class UnifiedLLM:
    """
    A unified interface that abstracts differences between HF transformer models
    and llama_cpp models.

    backend: "hf" for Hugging Face models, "llama_cpp" for llama_cpp models.
    model: The underlying model object.
    tokenizer: (Optional) The tokenizer (used only for HF backend).
    """

    def __init__(self, backend: str, model: Any, tokenizer: Any | None = None):
        """
        Initialize the UnifiedLLM with a backend, model, and optional tokenizer.

        Args:
            backend: "hf" for Hugging Face models, "llama_cpp" for llama_cpp models
            model: The underlying model object
            tokenizer: The tokenizer object (only needed for HF backend)
        """
        self.backend = backend.lower()
        self.model = model

        if self.backend == "hf":
            # Ensure the HF model is in evaluation mode.
            self.model.eval()
            self.tokenizer = tokenizer
        elif self.backend == "llama_cpp":
            # For llama_cpp, we still need a tokenizer-like interface
            self.tokenizer = self  # Point to self so we can use unified tokenization methods
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        if self.backend == "hf":
            generation_config = kwargs.get(
                "generation_config",
                {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "pad_token_id": self.tokenizer.eos_token_id,
                },
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", **kwargs)
            output = self.model.generate(**inputs, generation_config=generation_config)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif self.backend == "llama_cpp":
            generate_kwargs = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
            }
            response = self.model.create_completion(**generate_kwargs)
            return response["choices"][0]["text"]
        else:
            raise ValueError("Unsupported backend")

    def tokenize(self, text: str) -> Any:
        """
        Tokenize text. For HF models this uses the provided tokenizer; for llama_cpp,
        tokenization is handled internally (if available).

        Args:
            text: The text to tokenize

        Returns:
            Tokenized representation
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

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> dict:
        """
        Implement the __call__ method to emulate HF tokenizer behavior.
        This is crucial for compatibility with code that calls tokenizer directly.

        Args:
            text: The text to tokenize
            return_tensors: Format of return tensors ("pt" for PyTorch)
            **kwargs: Additional tokenization parameters

        Returns:
            A dict-like object with tokenized representations
        """
        if self.backend == "hf":
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        elif self.backend == "llama_cpp":
            # Create a custom dict subclass to support a .to() method.
            class TokenizerOutput(dict):
                def to(self, device):
                    # llama_cpp doesn't actually use the device, so simply return self
                    return self

            # Minimal implementation for llama_cpp tokenization.
            if hasattr(self.model, "tokenize"):
                # Encode the text to bytes (UTF-8) before tokenization
                tokens = self.model.tokenize(text.encode("utf-8"), add_bos=True, special=True)

                class TokenTensor:
                    def __init__(self, tokens):
                        self.tokens = tokens

                    def to(self, device):
                        # Just return self since llama_cpp doesn't use devices
                        return self

                # Wrap the tokens in our custom dict subclass.
                result = TokenizerOutput({"input_ids": TokenTensor(tokens)})
                return result
            else:
                # Fallback dummy output if no tokenization is available.
                class DummyTokenizerOutput(dict):
                    def to(self, device):
                        return self

                return DummyTokenizerOutput()
        else:
            raise ValueError("Unsupported backend")

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """
        Decode tokens to text. For HF models this uses the provided tokenizer;
        for llama_cpp, we implement a minimal version.

        Args:
            token_ids: The token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            The decoded text
        """
        if self.backend == "hf":
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        elif self.backend == "llama_cpp":
            # llama_cpp models don't typically need to decode separately
            # as generation returns text directly, but we provide this for API compatibility
            if hasattr(self.model, "detokenize"):
                return self.model.detokenize(token_ids)
            # For most cases with llama_cpp this won't be called, but we need
            # to provide it for compatibility
            return "[DECODED TEXT]"
        else:
            raise ValueError("Unsupported backend")

    # Add properties to emulate other tokenizer properties
    @property
    def eos_token_id(self):
        """Get the end-of-sequence token ID."""
        if self.backend == "hf":
            return self.tokenizer.eos_token_id
        elif self.backend == "llama_cpp":
            # Many llama models use token ID 2 as EOS
            return getattr(self.model, "eos_token_id", 2)
        else:
            raise ValueError("Unsupported backend")

    @property
    def pad_token_id(self):
        """Get the padding token ID."""
        if self.backend == "hf":
            return self.tokenizer.pad_token_id
        elif self.backend == "llama_cpp":
            # Often same as EOS for llama models
            return getattr(self.model, "pad_token_id", self.eos_token_id)
        else:
            raise ValueError("Unsupported backend")

    @property
    def device(self):
        """Get the device that the model is running on."""
        if self.backend == "hf":
            return self.model.device
        elif self.backend == "llama_cpp":
            # llama_cpp doesn't have a device concept in the same way
            # Return "cpu" as a default
            return "cpu"
        else:
            raise ValueError("Unsupported backend")
