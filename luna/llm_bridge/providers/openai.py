"""OpenAI provider — GPT / o-series API via the openai SDK."""

from __future__ import annotations

import os

from luna.llm_bridge.bridge import LLMBridge, LLMBridgeError, LLMResponse

# Reasoning models that require special handling:
# - No temperature parameter
# - max_tokens → max_completion_tokens
# - system role → developer role
_REASONING_MODELS = frozenset({
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
})


class OpenAIProvider(LLMBridge):
    """GPT / o-series API provider using ``openai.AsyncOpenAI``.

    Reasoning models (o1, o3, o4-mini) are auto-detected:
    temperature is omitted, system prompt uses the ``developer`` role,
    and ``max_completion_tokens`` replaces ``max_tokens``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise LLMBridgeError(
                "No OpenAI API key: set api_key or OPENAI_API_KEY env var.",
                provider="openai",
            )
        try:
            import openai
        except ImportError as exc:
            raise LLMBridgeError(
                "openai package not installed: pip install openai",
                provider="openai",
                original=exc,
            ) from exc
        kwargs: dict = {"api_key": self._api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._is_reasoning = self._model in _REASONING_MODELS

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            # Reasoning models use "developer" role instead of "system".
            role = "developer" if self._is_reasoning else "system"
            full_messages.append({"role": role, "content": system_prompt})
        full_messages.extend(messages)

        params: dict = {
            "model": self._model,
            "messages": full_messages,
        }

        if self._is_reasoning:
            # Reasoning models: no temperature, use max_completion_tokens.
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature

        try:
            response = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise LLMBridgeError(
                f"OpenAI API error: {exc}",
                provider="openai",
                original=exc,
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.close()
