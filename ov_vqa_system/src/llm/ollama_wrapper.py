

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a factual reasoning agent. "
    "Your task is to answer visual questions based strictly on provided evidence. "
    "Do not introduce unsupported information."
)


REASONING_SYSTEM_INSTRUCTION = (
    "You are a factual visual question answering agent. "
    "You must answer the question based strictly on the provided JSON evidence. "
    "The evidence may include visual detections and retrieved knowledge. "
    "Do not use unsupported assumptions. "
    "Return a short final answer."
)


CONCEPT_SYSTEM_INSTRUCTION = (
    "You are a visual concept extraction agent. "
    "Your task is to convert a visual question into short, concrete, detectable visual concepts. "
    "Return only a concise list of phrases."
)


@dataclass
class LLMResponse:
    """
    Structured LLM response.
    """

    text: str
    raw: Dict[str, Any]
    model_name: str


class OllamaWrapper:
    """
    Local Ollama wrapper.

    This class uses the local Ollama HTTP API:
        /api/generate

    It is intentionally simple and stable for long-running experiments.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 150,
        timeout: int = 60,
        stop: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        raise_on_error: bool = True,
    ) -> None:
        """
        Args:
            model_name: Local Ollama model name, e.g. llama3, llama3:8b.
            base_url: Ollama base URL.
            temperature: Default sampling temperature.
            top_p: Default nucleus sampling value.
            max_tokens: Default max generation length.
            timeout: HTTP request timeout in seconds.
            stop: Optional stop strings.
            system_instruction: Default system instruction.
            raise_on_error:
                If True, raise RuntimeError when Ollama fails.
                If False, return an error string.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stop = stop or ["User Input:", "Instruction:"]
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.raise_on_error = raise_on_error

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "OllamaWrapper":
        """
        Build wrapper from config.

        Args:
            cfg: Full experiment config.

        Returns:
            OllamaWrapper.
        """
        llm_cfg = cfg.get("llm", {})

        return cls(
            model_name=llm_cfg.get("model_name", "llama3"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            top_p=float(llm_cfg.get("top_p", 0.9)),
            max_tokens=int(llm_cfg.get("max_tokens", 150)),
            timeout=int(llm_cfg.get("timeout", 60)),
            stop=llm_cfg.get("stop", ["User Input:", "Instruction:"]),
            system_instruction=llm_cfg.get(
                "system_instruction",
                DEFAULT_SYSTEM_INSTRUCTION,
            ),
            raise_on_error=bool(llm_cfg.get("raise_on_error", True)),
        )

    def build_prompt(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Combine system instruction and user prompt.

        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction override.

        Returns:
            Full prompt string.
        """
        instruction = system_instruction or self.system_instruction

        return (
            f"{instruction}\n\n"
            f"User Input:\n"
            f"{prompt}"
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
        return_response: bool = False,
    ) -> str | LLMResponse:
        """
        Generate text with Ollama.

        Args:
            prompt: User prompt.
            temperature: Optional temperature override.
            top_p: Optional top_p override.
            max_tokens: Optional generation length override.
            stop: Optional stop sequence override.
            system_instruction: Optional system instruction override.
            return_response: Whether to return structured response.

        Returns:
            Generated text or LLMResponse.
        """
        full_prompt = self.build_prompt(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p if top_p is None else top_p,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                "stop": self.stop if stop is None else stop,
            },
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            raw = response.json()
            text = raw.get("response", "").strip()

            llm_response = LLMResponse(
                text=text,
                raw=raw,
                model_name=self.model_name,
            )

            if return_response:
                return llm_response

            return text

        except Exception as e:
            message = (
                f"Ollama generation failed. "
                f"model={self.model_name}, url={self.api_url}, error={str(e)}"
            )

            if self.raise_on_error:
                raise RuntimeError(message) from e

            return f"Reasoning Error: {message}"

    def generate_concepts(
        self,
        prompt: str,
        max_tokens: int = 200,
    ) -> str:
        """
        Generate concept list.

        Args:
            prompt: Concept generation prompt.
            max_tokens: Generation length.

        Returns:
            Raw LLM output.
        """
        return self.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            system_instruction=CONCEPT_SYSTEM_INSTRUCTION,
        )

    def generate_reasoning(
        self,
        prompt: str,
        max_tokens: int = 150,
    ) -> str:
        """
        Generate final reasoning answer.

        Args:
            prompt: Evidence-based reasoning prompt.
            max_tokens: Generation length.

        Returns:
            Raw LLM output.
        """
        return self.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            system_instruction=REASONING_SYSTEM_INSTRUCTION,
        )

    def check_available(self) -> bool:
        """
        Check whether local Ollama server is available.

        Returns:
            True if Ollama is reachable.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List local Ollama models.

        Returns:
            List of model names.
        """
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()

        models = []
        for item in data.get("models", []):
            name = item.get("name")
            if name:
                models.append(name)

        return models

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """
        Alias for generate().
        """
        return self.generate(prompt, **kwargs)


def build_llm(cfg: Dict[str, Any]) -> OllamaWrapper:
    """
    Build LLM from config.

    Args:
        cfg: Full experiment config.

    Returns:
        OllamaWrapper.
    """
    llm_cfg = cfg.get("llm", {})
    provider = str(llm_cfg.get("provider", "ollama")).lower()

    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Currently only ollama is implemented."
        )

    return OllamaWrapper.from_config(cfg)