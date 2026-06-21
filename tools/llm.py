"""
Shared Groq LLM client with LangSmith tracing.

All LLM calls in this project go through `chat()`.
LangSmith traces every call automatically when LANGCHAIN_TRACING_V2=true.
"""
from __future__ import annotations

from groq import Groq
from langsmith import traceable

import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


@traceable(run_type="llm", name="groq-chat")
def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = 2000,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    """
    Call Groq chat completion and return the assistant text.

    Args:
        messages:    OpenAI-format message list.
        model:       Groq model ID; defaults to config.LLM_MODEL.
        max_tokens:  Max output tokens.
        json_mode:   When True, forces JSON object output (use for JD parsing).
        temperature: Sampling temperature (lower = more deterministic).
    """
    kwargs: dict = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = _get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()
