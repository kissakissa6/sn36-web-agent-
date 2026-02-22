"""LLM gateway client for the Autoppia sandbox.

All requests go through OPENAI_BASE_URL (injected by the validator sandbox
as http://sandbox-gateway:9000/openai/v1). Every request carries the
IWA-Task-ID header for tracking and billing.
"""

from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


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


class _TransientError(Exception):
    """Wrapper for transient errors that should be retried."""
    pass


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception_type((_TransientError,)),
    reraise=True,
)
async def _do_request(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Make the HTTP request with retry logic for transient failures."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.RequestError as exc:
            raise _TransientError(str(exc)) from exc

        if resp.status_code >= 500:
            raise _TransientError(f"Server error {resp.status_code}: {resp.text[:200]}")

        resp.raise_for_status()
        return resp.json()


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
    max_tokens = max_tokens if max_tokens is not None else _MAX_TOKENS

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

    try:
        data = await _do_request(url, headers, body)
    except _TransientError as exc:
        logger.error(f"LLM transient error (after retries): {exc}")
        raise
    except httpx.HTTPStatusError as exc:
        error_text = exc.response.text[:500] if exc.response else str(exc)
        logger.error(f"LLM API error ({exc.response.status_code}): {error_text}")

        # Retry without response_format if unsupported (400 error)
        if response_format and exc.response and exc.response.status_code == 400:
            logger.warning("Retrying without response_format")
            body.pop("response_format", None)
            try:
                data = await _do_request(url, headers, body)
            except Exception:
                raise
        else:
            raise

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "content": message.get("content", ""),
        "usage": data.get("usage", {}),
        "model": data.get("model", model),
        "finish_reason": choice.get("finish_reason", ""),
    }


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM content, handling various formats."""
    content = content.strip()
    if not content:
        return None

    # Try direct parse first
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    if "```" in content:
        for block in content.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    result = json.loads(block)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue

    # Try to find a JSON object in the content using regex
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


async def get_action_decision(
    messages: List[Dict[str, str]],
    task_id: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Call LLM and parse the JSON action decision.

    Returns the parsed dict or an empty dict on failure.
    """
    try:
        result = await chat_completion(
            messages=messages,
            task_id=task_id,
            model=model,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return {}

    content = result.get("content", "").strip()
    if not content:
        return {}

    parsed = _extract_json(content)
    if parsed is None:
        logger.warning(f"Failed to parse LLM JSON: {content[:200]}")
        return {}

    return parsed
