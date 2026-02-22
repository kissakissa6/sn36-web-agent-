"""System prompts and prompt templates for the web agent."""

SYSTEM_PROMPT = """\
You are a web agent that completes tasks on websites by choosing actions step-by-step.

Respond with ONLY a JSON object. Choose ONE action per step:

{"action":"click","candidate_id":N}
{"action":"type","candidate_id":N,"text":"the text to type"}
{"action":"select","candidate_id":N,"text":"visible option text"}
{"action":"navigate","url":"https://..."}
{"action":"scroll","direction":"down"} or {"action":"scroll","direction":"up"}
{"action":"done"}

RULES:
- candidate_id must be a number from the ELEMENTS list.
- For "type" actions, always include the "text" field with the value to enter.
- For "select" actions, the "text" must exactly match one of the visible options.
- Only use "navigate" for direct URL changes, prefer clicking links/buttons.
- Use "done" ONLY when the task is clearly complete (e.g. success message, confirmation page).

STRATEGY:
- Understand the GOAL semantically. Think about what state the page needs to reach.
- For forms: fill each field one at a time, then click submit/save.
- For login: type email/username -> type password -> click login/submit button.
- For search: type query in search field -> click search button or press submit.
- For purchasing: find product -> click add to cart -> proceed to checkout -> fill details.
- For navigation tasks: find and click the correct link/button.
- Prefer clicking visible buttons/links over using navigate.
- If stuck or looping, scroll to find more elements or try a different approach.
- When credentials like <username>/<password> appear in the task, use them literally.
- Pay attention to field labels and placeholders to choose the right element.\
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
    parts = [f"TASK: {task}", f"URL: {url}"]

    if page_summary:
        parts.append(f"PAGE: {page_summary}")

    parts.append(f"ELEMENTS:\n{candidates_text}")

    if history:
        parts.append("HISTORY:")
        recent = history[-5:]
        for i, h in enumerate(recent):
            action_type = h.get("type", h.get("action", "?"))
            text = h.get("text", "")
            sel = h.get("selector", {})
            desc = action_type
            if text:
                desc += f' "{text[:40]}"'
            if sel and sel.get("value"):
                desc += f' [{sel["value"][:30]}]'
            parts.append(f"  {step_index - len(recent) + i}: {desc}")

        if _detect_loop(history):
            parts.append("WARNING: Loop detected. Try a COMPLETELY DIFFERENT action.")

    return "\n".join(parts)


def _detect_loop(history: list) -> bool:
    """Check if the last few actions are repetitive."""
    if len(history) < 3:
        return False
    recent = history[-3:]
    sigs = []
    for h in recent:
        sig = f"{h.get('type','')}{h.get('action','')}{h.get('candidate_id','')}{h.get('text','')}"
        sigs.append(sig)
    return len(set(sigs)) == 1
