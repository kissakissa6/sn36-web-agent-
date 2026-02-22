"""HTML parsing and interactive element extraction.

Parses snapshot_html to find actionable elements (buttons, links, inputs,
selects, textareas) and builds a compact representation for the LLM prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup, Tag

from actions import make_selector


# Tags considered interactive
_INTERACTIVE_TAGS = {"a", "button", "input", "textarea", "select", "option"}
_CLICKABLE_ROLES = {"button", "link", "tab", "menuitem", "checkbox", "radio", "switch"}

# Maximum candidates to send to LLM (keep prompt small for speed)
MAX_CANDIDATES = 40


def parse_html(html: str) -> List[Dict[str, Any]]:
    """Extract interactive elements from HTML and return candidate list.

    Each candidate is a dict with:
      - id: int (index in the list)
      - tag: str
      - text: str (visible label)
      - type: str (input type or element type)
      - selector: dict (IWA selector format)
      - attributes: dict (relevant attrs)
      - context: str (parent context)
    """
    if not html or not html.strip():
        return []

    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, and hidden elements
    for tag in soup.find_all(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()

    candidates: List[Dict[str, Any]] = []
    seen_selectors: set = set()

    # 1. Collect explicitly interactive elements
    for element in soup.find_all(_INTERACTIVE_TAGS):
        candidate = _extract_candidate(element)
        if candidate and candidate["_key"] not in seen_selectors:
            seen_selectors.add(candidate["_key"])
            candidates.append(candidate)

    # 2. Collect elements with interactive roles
    for element in soup.find_all(attrs={"role": True}):
        role = element.get("role", "").lower()
        if role in _CLICKABLE_ROLES:
            candidate = _extract_candidate(element)
            if candidate and candidate["_key"] not in seen_selectors:
                seen_selectors.add(candidate["_key"])
                candidates.append(candidate)

    # 3. Collect clickable elements (onclick, tabindex, contenteditable)
    for element in soup.find_all(attrs={"onclick": True}):
        candidate = _extract_candidate(element)
        if candidate and candidate["_key"] not in seen_selectors:
            seen_selectors.add(candidate["_key"])
            candidates.append(candidate)

    # Prioritize: form inputs first, then buttons, then links
    candidates.sort(key=_candidate_priority)

    # Trim to max and assign sequential IDs
    candidates = candidates[:MAX_CANDIDATES]
    for i, c in enumerate(candidates):
        c["id"] = i
        c.pop("_key", None)
        c.pop("_priority", None)

    return candidates


def build_page_summary(html: str) -> str:
    """Build a short text summary of the page for context."""
    if not html:
        return "Empty page."

    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # Get title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Get headings
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"], limit=5):
        text = h.get_text(strip=True)[:80]
        if text:
            headings.append(f"  {h.name}: {text}")

    # Get forms
    forms = []
    for form in soup.find_all("form", limit=3):
        inputs = form.find_all(["input", "textarea", "select"])
        labels = [_get_input_label(inp) for inp in inputs[:5]]
        labels = [l for l in labels if l]
        if labels:
            forms.append(f"  Form fields: {', '.join(labels)}")

    parts = []
    if title:
        parts.append(f"Page: {title}")
    if headings:
        parts.append("Headings:\n" + "\n".join(headings))
    if forms:
        parts.append("Forms:\n" + "\n".join(forms))

    return "\n".join(parts) if parts else "Page with interactive elements."


def format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Format candidates as a concise numbered list for the LLM prompt."""
    if not candidates:
        return "No interactive elements found on this page."

    lines = []
    for c in candidates:
        tag = c["tag"]
        text = c["text"][:60] if c["text"] else ""
        elem_type = c.get("type", "")
        attrs = c.get("attributes", {})
        context = c.get("context", "")

        # Build a readable description
        parts = [f"[{c['id']}]"]

        if tag == "input":
            input_type = elem_type or "text"
            placeholder = attrs.get("placeholder", "")
            name = attrs.get("name", "")
            value = attrs.get("value", "")
            label = text or placeholder or name or input_type
            desc = f"<input type={input_type}> {label}"
            if value:
                desc += f' (current: "{value[:30]}")'
            parts.append(desc)
        elif tag == "textarea":
            placeholder = attrs.get("placeholder", "")
            parts.append(f"<textarea> {text or placeholder}")
        elif tag == "select":
            options = c.get("options", [])
            opts_str = ", ".join(options[:5])
            if opts_str:
                parts.append(f"<select> {text or attrs.get('name','')} options=[{opts_str}]")
            else:
                parts.append(f"<select> {text}")
        elif tag == "a":
            href = attrs.get("href", "")
            parts.append(f"<a> \"{text}\"")
            if href and not href.startswith("javascript:"):
                parts.append(f"href={href[:60]}")
        elif tag == "button":
            parts.append(f"<button> \"{text}\"")
        else:
            parts.append(f"<{tag}> \"{text}\"")

        if context:
            parts.append(f"[{context}]")

        lines.append(" ".join(parts))

    return "\n".join(lines)


# --- Internal helpers ---


def _extract_candidate(element: Tag) -> Optional[Dict[str, Any]]:
    """Extract a candidate dict from a BeautifulSoup element."""
    tag = element.name
    if not tag:
        return None

    text = _get_visible_text(element)
    elem_type = element.get("type", "")

    # Skip hidden/disabled inputs
    if tag == "input" and elem_type in ("hidden",):
        return None
    if element.get("disabled") is not None:
        return None
    # Skip empty, meaningless elements
    if tag == "option":
        return None  # options are part of select, not standalone

    # Build selector - prefer id, then name, then text content
    selector = _build_selector(element)
    if not selector:
        return None

    # Get relevant attributes
    attributes: Dict[str, str] = {}
    for attr in ("name", "placeholder", "href", "value", "aria-label", "title"):
        val = element.get(attr)
        if val and isinstance(val, str):
            attributes[attr] = val[:100]

    # Get parent context
    context = _get_parent_context(element)

    # For select elements, extract options
    options: list = []
    if tag == "select":
        for opt in element.find_all("option", limit=10):
            opt_text = opt.get_text(strip=True)
            if opt_text:
                options.append(opt_text[:40])

    # Get associated label text
    label_text = _get_associated_label(element)
    if label_text and not text:
        text = label_text

    # Dedup key
    key = f"{tag}:{selector.get('value', '')}:{text[:30]}"

    result = {
        "tag": tag,
        "text": text,
        "type": str(elem_type),
        "selector": selector,
        "attributes": attributes,
        "context": context,
        "_key": key,
        "_priority": _candidate_priority_value(tag, elem_type),
    }
    if options:
        result["options"] = options
    return result


def _build_selector(element: Tag) -> Optional[Dict[str, Any]]:
    """Build the best available selector for an element."""
    # Prefer id attribute
    elem_id = element.get("id")
    if elem_id and isinstance(elem_id, str) and elem_id.strip():
        return make_selector(attribute="id", value=elem_id.strip())

    # Then data-testid
    testid = element.get("data-testid")
    if testid and isinstance(testid, str):
        return make_selector(attribute="data-testid", value=testid.strip())

    # Then name attribute
    name = element.get("name")
    if name and isinstance(name, str) and name.strip():
        return make_selector(attribute="name", value=name.strip())

    # Then href for links
    if element.name == "a":
        href = element.get("href")
        if href and isinstance(href, str) and not href.startswith("javascript:"):
            return make_selector(attribute="href", value=href.strip())

    # Fall back to text content matching
    text = _get_visible_text(element)
    if text:
        return make_selector(
            selector_type="tagContainsSelector",
            attribute="text",
            value=text[:80],
        )

    return None


def _get_visible_text(element: Tag) -> str:
    """Get visible text content of an element, trimmed."""
    text = element.get_text(separator=" ", strip=True)
    # Also check aria-label and title
    if not text:
        text = element.get("aria-label", "") or element.get("title", "")
        if isinstance(text, list):
            text = " ".join(text)
    return str(text)[:120].strip()


def _get_parent_context(element: Tag) -> str:
    """Get a brief description of the element's parent context."""
    parent = element.parent
    if not parent or not isinstance(parent, Tag):
        return ""

    # Check for form, nav, header, footer, main, aside
    for ancestor in element.parents:
        if not isinstance(ancestor, Tag):
            continue
        if ancestor.name in ("form", "nav", "header", "footer", "main", "aside"):
            return ancestor.name
        role = ancestor.get("role", "")
        if role in ("navigation", "banner", "main", "form"):
            return role

    return ""


def _get_associated_label(element: Tag) -> str:
    """Find label text associated with an input element."""
    elem_id = element.get("id")
    if elem_id and element.find_parent():
        # Search the whole document for label[for=id]
        root = element.find_parent()
        while root and root.parent and isinstance(root.parent, Tag) and root.parent.name != "[document]":
            root = root.parent
        if root:
            label = root.find("label", attrs={"for": elem_id})
            if label:
                return label.get_text(strip=True)[:60]

    # Check if element is inside a label
    parent_label = element.find_parent("label")
    if parent_label:
        label_text = parent_label.get_text(strip=True)[:60]
        elem_text = element.get_text(strip=True)
        if label_text != elem_text:
            return label_text

    return ""


def _get_input_label(element: Tag) -> str:
    """Get a label for a form input."""
    # Check for associated label
    elem_id = element.get("id")
    if elem_id:
        label = element.find_parent().find("label", attrs={"for": elem_id}) if element.find_parent() else None
        if label:
            return label.get_text(strip=True)[:40]

    # Check placeholder, name, aria-label
    for attr in ("placeholder", "name", "aria-label"):
        val = element.get(attr)
        if val and isinstance(val, str):
            return val[:40]

    return element.get("type", "input")


def _candidate_priority(candidate: Dict[str, Any]) -> int:
    """Sort key: lower = higher priority."""
    return candidate.get("_priority", 50)


def _candidate_priority_value(tag: str, elem_type: str) -> int:
    """Priority value for sorting candidates."""
    if tag == "input" and elem_type in ("text", "email", "password", "search", "tel", "url"):
        return 10  # Text inputs first (form filling)
    if tag == "textarea":
        return 11
    if tag == "select":
        return 12
    if tag == "input" and elem_type in ("submit",):
        return 15  # Submit buttons
    if tag == "button":
        return 20
    if tag == "input" and elem_type in ("checkbox", "radio"):
        return 25
    if tag == "a":
        return 30  # Links last
    return 40
