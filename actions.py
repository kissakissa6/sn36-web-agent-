"""IWA-compatible action types for the web agent.

Action format reference (from autoppia_iwa):
- ClickAction: selector (optional), x, y
- TypeAction: text, selector (optional)
- SelectDropDownOptionAction: selector, text, timeout_ms
- NavigateAction: url, go_back, go_forward
- ScrollAction: value, up, down, left, right
- WaitAction: selector, time_seconds, timeout_seconds
- IdleAction: (no fields) - used for "done" / no-op
- SubmitAction: selector
- HoverAction: selector

Selector format: {type, attribute, value, case_sensitive}
Selector types: "attributeValueSelector", "tagContainsSelector", "xpathSelector"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger


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
    return {
        "type": "ScrollAction",
        "down": down,
        "up": not down,
        "left": False,
        "right": False,
    }


def wait_action(seconds: float = 1.0) -> Dict[str, Any]:
    return {"type": "WaitAction", "time_seconds": seconds}


def idle_action() -> Dict[str, Any]:
    """IWA IdleAction - used when the task appears complete (no DoneAction in IWA)."""
    return {"type": "IdleAction"}


def submit_action(selector: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "SubmitAction", "selector": selector}


def hover_action(selector: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "HoverAction", "selector": selector}


def _normalize_action_name(raw: str) -> str:
    """Normalize an action name from LLM output to a canonical form.

    Handles cases like:
      "ClickAction" -> "click"
      "SelectDropDownOptionAction" -> "selectdropdownoption"
      "scroll" -> "scroll"
      "type" -> "type"
    """
    name = raw.lower().strip()
    # Remove trailing "action" suffix if present
    if name.endswith("action"):
        name = name[:-6]
    return name.strip()


# Map of normalized action names to canonical action type
_ACTION_ALIASES = {
    "click": "click",
    "type": "type",
    "input": "type",
    "fill": "type",
    "select": "select",
    "selectdropdown": "select",
    "selectdropdownoption": "select",
    "dropdown": "select",
    "navigate": "navigate",
    "goto": "navigate",
    "go": "navigate",
    "scroll": "scroll",
    "scrolldown": "scroll_down",
    "scrollup": "scroll_up",
    "wait": "wait",
    "sleep": "wait",
    "pause": "wait",
    "done": "idle",
    "complete": "idle",
    "finish": "idle",
    "idle": "idle",
    "submit": "submit",
    "hover": "hover",
}


def build_action_from_llm(decision: Dict[str, Any], candidates: list) -> Optional[Dict[str, Any]]:
    """Convert LLM decision dict into a proper IWA action."""
    raw_action = decision.get("action", decision.get("type", ""))
    if not raw_action:
        return None

    normalized = _normalize_action_name(str(raw_action))
    action = _ACTION_ALIASES.get(normalized)

    # If not found in aliases, try partial matching
    if action is None:
        for alias, canonical in _ACTION_ALIASES.items():
            if alias in normalized or normalized in alias:
                action = canonical
                break

    if action is None:
        logger.warning(f"Unknown action type: {raw_action!r} (normalized: {normalized!r})")
        return None

    candidate_id = decision.get("candidate_id")

    # Also accept candidate_id as string
    if isinstance(candidate_id, str):
        try:
            candidate_id = int(candidate_id)
        except ValueError:
            candidate_id = None

    # Resolve selector from candidate list
    selector = None
    if candidate_id is not None and isinstance(candidate_id, int) and 0 <= candidate_id < len(candidates):
        selector = candidates[candidate_id].get("selector")

    if action == "click":
        if selector:
            return click_action(selector)
        # If LLM specified a selector directly in the decision
        if "selector" in decision and isinstance(decision["selector"], dict):
            return click_action(decision["selector"])
    elif action == "type":
        text = decision.get("text", decision.get("value", ""))
        if selector:
            # Allow typing empty string (to clear a field)
            return type_action(selector, str(text))
    elif action == "select":
        text = decision.get("text", decision.get("option", decision.get("value", "")))
        if selector and text:
            return select_option_action(selector, str(text))
    elif action == "navigate":
        url = decision.get("url", "")
        if url:
            return navigate_action(url)
    elif action in ("scroll", "scroll_down", "scroll_up"):
        direction = decision.get("direction", "down").lower()
        if action == "scroll_up" or direction == "up":
            return scroll_action(down=False)
        return scroll_action(down=True)
    elif action == "wait":
        try:
            seconds = float(decision.get("seconds", decision.get("time", decision.get("time_seconds", 1.0))))
        except (ValueError, TypeError):
            seconds = 1.0
        return wait_action(seconds)
    elif action == "idle":
        return idle_action()
    elif action == "submit":
        if selector:
            return submit_action(selector)
    elif action == "hover":
        if selector:
            return hover_action(selector)

    return None
