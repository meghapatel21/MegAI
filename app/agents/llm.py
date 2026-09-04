"""Single place that builds the chat model, so agents stay configuration-free."""

import json
import re

from app import config

_llm = None


def get_llm():
    """Return a shared ChatGroq instance, or None when no key is configured."""
    global _llm
    if not config.GROQ_API_KEY:
        return None
    if _llm is None:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    return _llm


def extract_json(raw):
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to, and a agent that
    dies on a stray backtick is worse than one that digs the object out.
    """
    if not raw:
        raise ValueError("empty response")

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON object in response: {raw[:200]}")


def invoke_json(llm, prompt, default=None):
    """Call the model and parse a JSON object, returning `default` on failure."""
    try:
        return extract_json(llm.invoke(prompt).content)
    except Exception as exc:
        print(f"[LLM] JSON parse failed ({exc}); using default.")
        return default
