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


def done_action() -> Dict[str, Any]:
    return {"type": "DoneAction"}


def build_action_from_llm(decision: Dict[str, Any], candidates: list) -> Optional[Dict[str, Any]]:
    """Convert LLM decision dict into a proper IWA action."""
    raw_action = decision.get("action", "")
    action = raw_action.lower().replace("action", "").strip()
    candidate_id = decision.get("candidate_id")

    # Also accept candidate_id as string
    if isinstance(candidate_id, str) and candidate_id.isdigit():
        candidate_id = int(candidate_id)

    # Resolve selector from candidate list
    selector = None
    if candidate_id is not None and isinstance(candidate_id, int) and 0 <= candidate_id < len(candidates):
        selector = candidates[candidate_id].get("selector")

    if action == "click":
        if selector:
            return click_action(selector)
    elif action in ("type", "input", "fill"):
        text = decision.get("text", decision.get("value", ""))
        if selector and text:
            return type_action(selector, text)
    elif action in ("select", "selectdropdown", "dropdown"):
        text = decision.get("text", decision.get("option", ""))
        if selector and text:
            return select_option_action(selector, text)
    elif action in ("navigate", "goto", "go"):
        url = decision.get("url", "")
        if url:
            return navigate_action(url)
    elif action in ("scroll", "scrolldown", "scrollup"):
        direction = decision.get("direction", "down").lower()
        if "up" in action:
            direction = "up"
        return scroll_action(down=(direction != "up"))
    elif action in ("wait", "sleep", "pause"):
        seconds = float(decision.get("seconds", decision.get("time", 1.0)))
        return wait_action(seconds)
    elif action in ("done", "complete", "finish"):
        return done_action()

    return None
