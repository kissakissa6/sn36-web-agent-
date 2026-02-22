"""LLM gateway client for the Autoppia sandbox.

All requests go through OPENAI_BASE_URL (injected by the validator sandbox
as http://sandbox-gateway:9000/openai/v1). Every request carries the
IWA-Task-ID header for tracking and billing.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger


_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-placeholder")
_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "256"))
_TIMEOUT: float = 30.0


def _is_sandbox() -> bool:
    """Detect if running inside the validator sandbox."""
    url = _BASE_URL.lower()
    return any(h in url for h in ("sandbox-gateway", "localhost", "127.0.0.1"))


async def chat_completion(
    messages: List[Dict[str, str]],
    task_id: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send a chat completion request through the gateway.

    Returns the parsed response dict with at least:
      - content: str (the assistant message)
      - usage: dict (token counts)
    """
    model = model or _MODEL
    temperature = temperature if temperature is not None else _TEMPERATURE
    max_tokens = max_tokens or _MAX_TOKENS

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "IWA-Task-ID": str(task_id),
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = response_format

    url = f"{_BASE_URL.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text[:500] if exc.response else str(exc)
            logger.error(f"LLM API error ({exc.response.status_code}): {error_text}")

            # Retry without response_format if unsupported
            if response_format and exc.response and exc.response.status_code == 400:
                logger.warning("Retrying without response_format")
                body.pop("response_format", None)
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
            else:
                raise
        except httpx.RequestError as exc:
            logger.error(f"LLM request failed: {exc}")
            raise

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "content": message.get("content", ""),
        "usage": data.get("usage", {}),
        "model": data.get("model", model),
        "finish_reason": choice.get("finish_reason", ""),
    }


async def get_action_decision(
    messages: List[Dict[str, str]],
    task_id: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Call LLM and parse the JSON action decision.

    Returns the parsed dict or an empty dict on failure.
    """
    result = await chat_completion(
        messages=messages,
        task_id=task_id,
        model=model,
        response_format={"type": "json_object"},
    )

    content = result.get("content", "").strip()
    if not content:
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```" in content:
            for block in content.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        logger.warning(f"Failed to parse LLM JSON: {content[:200]}")
        return {}
