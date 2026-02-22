"""IWA-compatible action types for the web agent."""

from __future__ import annotations

from typing import Any, Dict, Optional


def make_selector(
    *,
    selector_type: str = "attributeValueSelector",
    attribute: str = "id",
    value: str = "",
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Build a selector dict matching IWA format."""
    return {
        "type": selector_type,
        "attribute": attribute,
        "value": value,
        "case_sensitive": case_sensitive,
    }


def click_action(selector: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "ClickAction", "selector": selector}


def type_action(selector: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {"type": "TypeAction", "selector": selector, "text": text}


def select_option_action(selector: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {"type": "SelectDropDownOptionAction", "selector": selector, "text": text}


def navigate_action(url: str) -> Dict[str, Any]:
    return {"type": "NavigateAction", "url": url}


def scroll_action(*, down: bool = True) -> Dict[str, Any]:
    return {"type": "ScrollAction", "down": down, "up": not down}


def wait_action(seconds: float = 1.0) -> Dict[str, Any]:
    return {"type": "WaitAction", "time_seconds": seconds}


# Map of action type names to builder helpers (used by agent)
ACTION_TYPES = {
    "click", "type", "select", "navigate", "scroll", "wait",
    "ClickAction", "TypeAction", "SelectDropDownOptionAction",
    "NavigateAction", "ScrollAction", "WaitAction",
}


def build_action_from_llm(decision: Dict[str, Any], candidates: list) -> Optional[Dict[str, Any]]:
    """Convert LLM decision dict into a proper IWA action.

    The LLM is expected to return JSON like:
      {"action": "click", "candidate_id": 3}
      {"action": "type", "candidate_id": 5, "text": "hello"}
      {"action": "select", "candidate_id": 7, "text": "Option A"}
      {"action": "navigate", "url": "https://..."}
      {"action": "scroll", "direction": "down"}
      {"action": "wait", "seconds": 2}
    """
    action = decision.get("action", "").lower().replace("action", "").strip()
    candidate_id = decision.get("candidate_id")

    # Resolve selector from candidate list
    selector = None
    if candidate_id is not None and 0 <= candidate_id < len(candidates):
        selector = candidates[candidate_id].get("selector")

    if action == "click":
        if selector:
            return click_action(selector)
    elif action == "type":
        text = decision.get("text", "")
        if selector and text:
            return type_action(selector, text)
    elif action == "select":
        text = decision.get("text", "")
        if selector and text:
            return select_option_action(selector, text)
    elif action == "navigate":
        url = decision.get("url", "")
        if url:
            return navigate_action(url)
    elif action == "scroll":
        direction = decision.get("direction", "down").lower()
        return scroll_action(down=(direction != "up"))
    elif action == "wait":
        seconds = float(decision.get("seconds", 1.0))
        return wait_action(seconds)

    return None
