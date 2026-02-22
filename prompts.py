"""System prompts and prompt templates for the web agent."""

SYSTEM_PROMPT = """\
You are a web automation agent. You interact with web pages to complete tasks.

You receive:
- A task description (what to accomplish)
- The current page URL
- Interactive elements on the page (numbered list)
- Your action history (what you did so far)

You must respond with a single JSON object choosing ONE action:

ACTIONS:
1. Click an element: {"action": "click", "candidate_id": <number>}
2. Type into a field: {"action": "type", "candidate_id": <number>, "text": "<text>"}
3. Select dropdown option: {"action": "select", "candidate_id": <number>, "text": "<option text>"}
4. Navigate to URL: {"action": "navigate", "url": "<url>"}
5. Scroll the page: {"action": "scroll", "direction": "down"|"up"}
6. Wait for page load: {"action": "wait", "seconds": 1}

RULES:
- Respond with ONLY a JSON object, no other text.
- Choose exactly ONE action per response.
- Use candidate_id to reference elements from the numbered list.
- For login/signup: fill username/email first, then password, then click submit.
- For search: type query into search input, then click search button or press enter.
- For navigation: prefer clicking links over using navigate action.
- If a form has multiple fields, fill them one at a time (one field per action).
- If the page seems stuck or you're repeating actions, try a different approach.
- When credentials are shown as <username> and <password>, use them exactly as-is.\
"""


def build_user_prompt(
    task: str,
    url: str,
    page_summary: str,
    candidates_text: str,
    history: list,
    step_index: int,
) -> str:
    """Build the user message for the LLM."""
    parts = []

    parts.append(f"TASK: {task}")
    parts.append(f"URL: {url}")
    parts.append(f"STEP: {step_index}")

    if page_summary:
        parts.append(f"\nPAGE:\n{page_summary}")

    parts.append(f"\nELEMENTS:\n{candidates_text}")

    if history:
        parts.append("\nHISTORY (recent actions):")
        # Show last 6 actions
        recent = history[-6:] if len(history) > 6 else history
        for i, h in enumerate(recent):
            action_type = h.get("type", h.get("action", "unknown"))
            text = h.get("text", "")
            url_val = h.get("url", "")
            desc = action_type
            if text:
                desc += f' "{text}"'
            if url_val:
                desc += f" -> {url_val}"
            parts.append(f"  Step {step_index - len(recent) + i}: {desc}")

        # Detect loops
        if _detect_loop(history):
            parts.append("\nWARNING: You appear to be repeating actions. Try a DIFFERENT approach.")

    parts.append("\nRespond with a single JSON action:")

    return "\n".join(parts)


def _detect_loop(history: list) -> bool:
    """Check if the last few actions are repetitive."""
    if len(history) < 3:
        return False

    recent = history[-3:]
    # Check if all recent actions are identical
    signatures = []
    for h in recent:
        sig = f"{h.get('type', '')}{h.get('action', '')}{h.get('candidate_id', '')}{h.get('text', '')}"
        signatures.append(sig)

    return len(set(signatures)) == 1
