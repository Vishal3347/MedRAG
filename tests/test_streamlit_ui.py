"""Run the real Streamlit script headlessly (streamlit.testing) with the backend swapped for fakes."""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.test_ui_runtime import SETTINGS, FakeRetriever  # noqa: E402
from app.llm.mock import MockProvider  # noqa: E402
from app.llm.router import LLMRouter  # noqa: E402
from app.ui_runtime import Runtime  # noqa: E402


def test_page_answers_a_question_and_shows_sources(monkeypatch):
    import app.ui_runtime as ui

    monkeypatch.setattr(ui, "build_runtime", lambda settings: Runtime(FakeRetriever(), LLMRouter([MockProvider()], commit_chars=10), SETTINGS))
    at = AppTest.from_file("../deploy_streamlit/chat_ui.py", default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("How many chambers does the heart have?").run()
    assert not at.exception, at.exception
    assistant = [m for m in at.session_state.messages if m["role"] == "assistant"][0]
    assert assistant["status"] == "done" and "four chambers" in assistant["content"]
    assert assistant["sources"][0]["ref"] == "S1"
    assert any("Sources" in e.label for e in at.expander)
