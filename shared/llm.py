"""
LocalLLM - A simple wrapper around ollama.

This class provides a minimal interface to interact with local language models.
It intentionally has no magic:
- No retries (added in lesson 03)
- No tool calling (added in lesson 05)
- No memory (added in lesson 07)

Just text in, text out.
"""

import ollama

class LocalLLM:
    """
    A minimal wrapper for local LLM inference using Ollama.
    
    This class is intentionally simple and grows throughout the lessons.
    """
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        n_ctx: int = 2048
    ):
        """
        Initialize the local LLM.
        
        Args:
            model_path: Name of the Ollama model (e.g., "llama3.1")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate per response
            n_ctx: Context window size
        """
        self.model_name = model_path
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
    
    def generate(self, prompt: str, temperature: float = None, stop: list[str] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input text prompt
            temperature: Optional temperature override
            stop: Optional list of stop sequences
            
        Returns:
            Generated text as a string
        """
        options = {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.n_ctx,
            "stop": stop if stop is not None else ["</s>", "\n\n", "User:", "Assistant:"],
        }
        
        print(f"Generating with Ollama model '{self.model_name}' and options: {options}")
        print(f"Prompt: {prompt}")
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options
        )
        
        return response["response"].strip()