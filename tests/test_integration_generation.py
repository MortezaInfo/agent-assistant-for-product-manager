from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import src.ai.llm_client as llm_client
from src.ai.task_generator import generate_task_plan
from src.utils.errors import TaskGenerationError


def test_mock_mode_end_to_end_generation() -> None:
    plan = generate_task_plan("Build a CRM rollout plan", max_tasks=12, mode="mock")
    assert len(plan.tasks) >= 8
    assert plan.tasks[0].id == "T1"


def test_real_mode_missing_client_fails_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise RuntimeError("missing API key")

    monkeypatch.setattr("src.ai.task_generator.get_llm_client", _raise)

    with pytest.raises(TaskGenerationError) as exc_info:
        generate_task_plan("Build a CRM rollout plan", max_tasks=10, mode="real")

    assert "Task generation failed" in str(exc_info.value)


def test_llm_client_uses_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_options: dict[str, str] = {}

    def fake_openai(**options: str) -> object:
        captured_options.update(options)
        return object()

    monkeypatch.setattr(
        llm_client,
        "settings",
        SimpleNamespace(llm_api_key="test-key", llm_base_url="https://llm.example/v1"),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.setattr(llm_client, "_client", None)

    client = llm_client.get_llm_client()

    assert captured_options == {
        "api_key": "test-key",
        "base_url": "https://llm.example/v1",
    }
    assert client is llm_client.get_llm_client()
