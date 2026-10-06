"""Select one provider/key without exposing credentials or falling back silently."""

from .grok import Grok, OpenAI


def openai_key(values):
    return ((values.get("OPENAI_API_KEY") or "").strip()
            or (values.get("CHAT_GPT_KEY") or "").strip())


def create_model(client, values, *, provider=None, model=None, reasoning_effort=None,
                 search=None, history_images=None, state_mode="client"):
    provider = (provider or values.get("MODEL_PROVIDER") or "auto").strip().lower()
    if provider == "auto":
        provider = "openai" if openai_key(values) else "grok"
    if provider not in ("openai", "grok"):
        raise ValueError("MODEL_PROVIDER must be auto, openai or grok")
    prefix = "OPENAI" if provider == "openai" else "XAI"
    adapter = OpenAI if provider == "openai" else Grok
    return adapter(client, openai_key(values) if provider == "openai" else (values.get("XAI_API_KEY") or "").strip(),
        model=model or values.get(f"{prefix}_MODEL") or ("gpt-6.1-sol" if provider == "openai" else "grok-4.7"),
        reasoning_effort=reasoning_effort or values.get(f"{prefix}_REASONING_EFFORT") or ("low" if provider == "openai" else "medium"),
        search=search or values.get(f"{prefix}_SEARCH") or "auto",
        history_images=history_images or values.get(f"{prefix}_HISTORY_IMAGES") or "latest",
        state_mode=state_mode)
