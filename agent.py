"""Core web agent: FastAPI app with /health and /act endpoints.

Receives task context from the validator sandbox, parses the HTML snapshot,
calls the LLM to decide the next action, and returns an IWA-compatible action.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from loguru import logger

from actions import build_action_from_llm, click_action, scroll_action
from html_parser import format_candidates_for_prompt, parse_html, build_page_summary
from llm_client import get_action_decision
from prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

app = FastAPI(title="SN36 Web Agent", version="1.0.0")

# Optional metrics
_RETURN_METRICS = os.getenv("AGENT_RETURN_METRICS", "0") == "1"


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/act", summary="Decide next agent action")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Main endpoint called by the validator sandbox each step.

    Input payload:
        task_id: str
        prompt: str (or task_prompt)
        snapshot_html: str
        url: str
        step_index: int
        history: list[dict]
        model: str (optional override)
        screenshot: str|bytes|None (unused)
        target_hint: str (optional)

    Returns:
        {"actions": [<IWA action dict>]}
    """
    task_id = payload.get("task_id", "unknown")
    prompt = payload.get("prompt") or payload.get("task_prompt", "")
    snapshot_html = payload.get("snapshot_html", "")
    url = payload.get("url", "")
    step_index = payload.get("step_index", 0)
    history = payload.get("history") or []
    model_override = payload.get("model")
    target_hint = payload.get("target_hint", "")

    # Append target hint to prompt if available
    if target_hint and target_hint not in prompt:
        prompt = f"{prompt} (Hint: {target_hint})"

    logger.info(f"[{task_id}] Step {step_index} | URL: {url[:80]}")

    try:
        action = await _decide_action(
            task_id=task_id,
            prompt=prompt,
            snapshot_html=snapshot_html,
            url=url,
            step_index=step_index,
            history=history,
            model=model_override,
        )
    except Exception as exc:
        logger.error(f"[{task_id}] Agent error: {exc}")
        # Fallback: scroll down to reveal more content
        action = scroll_action(down=True)

    result: Dict[str, Any] = {"actions": [action]}

    if _RETURN_METRICS:
        result["metrics"] = {
            "task_id": task_id,
            "step_index": step_index,
            "url": url[:120],
        }

    return result


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


async def _decide_action(
    *,
    task_id: str,
    prompt: str,
    snapshot_html: str,
    url: str,
    step_index: int,
    history: list,
    model: str | None = None,
) -> Dict[str, Any]:
    """Parse HTML, build prompt, call LLM, return IWA action."""

    # 1. Parse HTML to extract interactive elements
    candidates = parse_html(snapshot_html)

    # 2. Build page summary
    page_summary = build_page_summary(snapshot_html)

    # 3. Format candidates for prompt
    candidates_text = format_candidates_for_prompt(candidates)

    # 4. Build messages
    user_prompt = build_user_prompt(
        task=prompt,
        url=url,
        page_summary=page_summary,
        candidates_text=candidates_text,
        history=history,
        step_index=step_index,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 5. Call LLM
    decision = await get_action_decision(
        messages=messages,
        task_id=task_id,
        model=model,
    )

    if not decision:
        logger.warning(f"[{task_id}] Empty LLM response, falling back to scroll")
        return scroll_action(down=True)

    # 6. Convert LLM decision to IWA action
    action = build_action_from_llm(decision, candidates)

    if not action:
        logger.warning(f"[{task_id}] Could not build action from: {decision}")
        # If LLM gave a candidate_id but action failed, try click as fallback
        cid = decision.get("candidate_id")
        if isinstance(cid, str):
            try:
                cid = int(cid)
            except ValueError:
                cid = None
        if cid is not None and isinstance(cid, int) and 0 <= cid < len(candidates):
            action = click_action(candidates[cid]["selector"])
        else:
            action = scroll_action(down=True)

    logger.info(f"[{task_id}] Action: {action.get('type', 'unknown')}")
    return action
