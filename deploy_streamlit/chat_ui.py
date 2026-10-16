"""Streamlit front end for MedRAG. Deploy on Streamlit Community Cloud with this file as the entrypoint."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the `app` package importable

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Ask the textbook", page_icon="📖", layout="centered")

# Streamlit Cloud secrets -> environment variables, so pydantic Settings picks them up.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(str(_k).upper(), str(_v))
except Exception:  # noqa: BLE001  (no secrets file when running locally with a .env)
    pass
os.environ.setdefault("LOG_JSON", "false")

from app.core.config import get_settings  # noqa: E402
from app.ui_runtime import build_runtime  # noqa: E402

MAX_QUESTIONS_PER_MINUTE = 6
HISTORY_TURNS = 8


@st.cache_resource(show_spinner="Loading the textbook index and models (first start takes a minute)…")
def get_runtime():
    return build_runtime(get_settings())


def citation_label(s: dict) -> str:
    pages = f"PDF p. {s['page_start']}" if s["page_start"] == s["page_end"] else f"PDF pp. {s['page_start']}–{s['page_end']}"
    title = (s.get("document_title") or "").strip()
    return f"{title}, {pages}" if title else pages


def render_sources(sources: list[dict], confident: bool | None) -> None:
    if not sources:
        return
    label = "Closest passages (low relevance)" if confident is False else f"Sources ({len(sources)})"
    with st.expander(label):
        for s in sources:
            where = " › ".join(x for x in (s.get("chapter"), s.get("section_heading")) if x)
            st.markdown(f"**{s['ref']}** · {citation_label(s)}" + (f"  \n*{where}*" if where else ""))
            st.caption(s["snippet"])


def rate_limited() -> float:
    """Per-session guard so one visitor cannot burn the shared free-tier quota. Returns seconds to wait."""
    now = time.time()
    stamps = [t for t in st.session_state.setdefault("stamps", []) if now - t < 60]
    st.session_state.stamps = stamps
    if len(stamps) >= MAX_QUESTIONS_PER_MINUTE:
        return 60 - (now - stamps[0])
    stamps.append(now)
    return 0.0


def history_for_model(messages: list[dict]) -> list[tuple[str, str]]:
    done = [m for m in messages if m["status"] == "done" and m.get("confident") is not False and m["content"].strip()]
    return [(m["role"], m["content"]) for m in done[-HISTORY_TURNS:]]


def answer(question: str, history: list[tuple[str, str]]) -> dict:
    rt = get_runtime()
    msg = {"role": "assistant", "content": "", "status": "streaming", "sources": [], "confident": None, "provider": None}
    body = st.empty()
    note = st.empty()
    text = ""
    for event, data in rt.stream(question, history):
        if event == "sources":
            msg["sources"], msg["confident"] = data["sources"], data["confident"]
        elif event == "provider":
            msg["provider"] = f"{data['provider']} · {data['model']}"
            note.empty()
        elif event == "token":
            text += data["text"]
            body.markdown(text + "▌")
        elif event == "restart":
            text = ""
            body.empty()
            note.caption(f"Switched to {data['to']} after {data['reason'].replace('_', ' ')}…")
        elif event == "done":
            msg["status"] = "done"
        elif event == "error":
            msg["status"] = "error"
            msg["error"] = data.get("message", "Something went wrong.")
    note.empty()
    body.markdown(text)
    msg["content"] = text
    if msg["status"] == "streaming":
        msg["status"] = "error"
        msg["error"] = "The connection closed before the answer finished."
    if msg["status"] == "error":
        st.error(msg["error"])
    render_sources(msg["sources"], msg["confident"])
    if msg["provider"]:
        st.caption(msg["provider"])
    return msg


# ------------------------------------------------------------------ page
st.title("📖 Ask the textbook")
st.caption("Answers come only from the indexed textbook and can still contain errors. Verify before any clinical use.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    with st.expander("Model status"):
        try:
            for p in get_runtime().provider_status():
                state = "ready" if p["available"] else f"cooling down {p['cooldown_remaining_s']:.0f}s"
                st.write(f"{p['provider']}/{p['model']}: {state}" + (f" (last error: {p['last_error']})" if p["last_error"] else ""))
        except Exception as exc:  # noqa: BLE001
            st.write(f"Not loaded yet ({type(exc).__name__}).")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("status") == "error":
                st.error(m.get("error", "Something went wrong."))
            render_sources(m.get("sources", []), m.get("confident"))
            if m.get("provider"):
                st.caption(m["provider"])

question = st.chat_input("Ask a question about the textbook")
if question:
    wait = rate_limited()
    if wait:
        st.warning(f"Too many questions in a short time. Please wait {wait:.0f}s.")
    else:
        history = history_for_model(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question.strip(), "status": "done"})
        with st.chat_message("user"):
            st.markdown(question.strip())
        with st.chat_message("assistant"):
            try:
                st.session_state.messages.append(answer(question.strip(), history))
            except Exception as exc:  # noqa: BLE001  (e.g. missing secrets -> clear message, not a stack trace)
                st.error(f"Could not start: {exc}")
                st.session_state.messages.pop()
