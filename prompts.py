"""System prompts and prompt templates for the web agent."""

SYSTEM_PROMPT = """\
You are a web agent that completes tasks on websites by choosing actions step-by-step.

Respond with ONLY a JSON object. Choose ONE action:

{"action":"click","candidate_id":N}
{"action":"type","candidate_id":N,"text":"..."}
{"action":"select","candidate_id":N,"text":"option text"}
{"action":"navigate","url":"..."}
{"action":"scroll","direction":"down"|"up"}
{"action":"done"}

STRATEGY:
- Understand the GOAL semantically. Think about what state the page needs to reach.
- For forms: fill each field one at a time, then click submit/save.
- For login: fill email/username -> fill password -> click login/submit.
- For search: type query -> click search or submit.
- For purchasing: find product -> add to cart -> proceed to checkout -> fill details.
- Prefer clicking visible buttons/links over navigate.
- If stuck or looping, scroll to find more elements or try a different approach.
- Use "done" when the task appears complete (e.g. confirmation message visible).
- When credentials like <username>/<password> appear in the task, use them literally.\
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
