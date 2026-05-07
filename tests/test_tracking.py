import pytest

import buc_factory.tracking as tracking
from buc_factory.tracking import (
    _MODEL_PRICING,
    _token_cost,
    log_task_result,
    reset_run,
)


@pytest.fixture(autouse=True)
def clean_task_log():
    reset_run()
    yield
    reset_run()


# ── _token_cost ───────────────────────────────────────────────────


def test_token_cost_opus():
    in_price, out_price = _MODEL_PRICING["claude-opus-4-7"]
    assert _token_cost(1_000_000, 1_000_000, "claude-opus-4-7") == in_price + out_price


def test_token_cost_sonnet():
    in_price, out_price = _MODEL_PRICING["claude-sonnet-4-6"]
    assert _token_cost(1_000_000, 1_000_000, "claude-sonnet-4-6") == in_price + out_price


def test_token_cost_haiku():
    in_price, out_price = _MODEL_PRICING["claude-haiku-4-5-20251001"]
    assert _token_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001") == in_price + out_price


def test_token_cost_gpt53():
    in_price, out_price = _MODEL_PRICING["gpt-5.3-chat-latest"]
    assert _token_cost(1_000_000, 1_000_000, "gpt-5.3-chat-latest") == in_price + out_price


def test_token_cost_o4_mini():
    in_price, out_price = _MODEL_PRICING["o4-mini"]
    assert _token_cost(1_000_000, 1_000_000, "o4-mini") == in_price + out_price


def test_token_cost_unknown_model_returns_zero():
    assert _token_cost(100, 50, "unknown-model") == 0.0


def test_token_cost_zero_tokens():
    assert _token_cost(0, 0, "claude-sonnet-4-6") == 0.0


def test_token_cost_proportional():
    assert _token_cost(2_000_000, 0, "claude-sonnet-4-6") == 2 * _token_cost(
        1_000_000, 0, "claude-sonnet-4-6"
    )


def test_openai_output_more_expensive_than_input():
    # For every listed OpenAI model, output tokens cost more than input tokens
    for model in ("gpt-5.3-chat-latest", "o4-mini"):
        in_price, out_price = _MODEL_PRICING[model]
        assert out_price > in_price, f"{model}: out_price should exceed in_price"


def test_claude_opus_more_expensive_than_haiku():
    opus_cost = _token_cost(1_000_000, 1_000_000, "claude-opus-4-7")
    haiku_cost = _token_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
    assert opus_cost > haiku_cost


# ── reset_run ─────────────────────────────────────────────────────


def test_reset_run_clears_task_log():
    tracking._task_log.append({"task": "dummy"})
    reset_run()
    assert tracking._task_log == []


def test_reset_run_on_empty_log_is_safe():
    reset_run()
    assert tracking._task_log == []


# ── log_task_result → _task_log ───────────────────────────────────


def test_log_task_result_success_appends_entry():
    log_task_result(
        "write_brief",
        ok=True,
        elapsed_s=12.5,
        retries=0,
        input_tokens=1000,
        output_tokens=500,
        model="claude-sonnet-4-6",
    )
    assert len(tracking._task_log) == 1
    e = tracking._task_log[0]
    assert e["task"] == "write_brief"
    assert e["status"] == "✓"
    assert e["attempts"] == 1  # retries + 1 when ok
    assert e["input_tokens"] == 1000
    assert e["output_tokens"] == 500
    assert e["cost"] > 0


def test_log_task_result_failure_appends_entry():
    log_task_result(
        "write_brief",
        ok=False,
        elapsed_s=5.0,
        retries=2,
        input_tokens=500,
        output_tokens=200,
        model="claude-sonnet-4-6",
    )
    e = tracking._task_log[0]
    assert e["status"] == "✗"
    assert e["attempts"] == 2  # retries (not +1) when not ok


def test_log_task_result_duration_format():
    log_task_result(
        "roll_scenario", ok=True, elapsed_s=90.0, retries=0, model="claude-haiku-4-5-20251001"
    )
    e = tracking._task_log[0]
    assert e["duration"] == "1m 30.0s"


def test_log_task_result_cost_uses_model_pricing():
    log_task_result(
        "t1",
        ok=True,
        elapsed_s=1.0,
        retries=0,
        input_tokens=1_000_000,
        output_tokens=0,
        model="claude-opus-4-7",
    )
    log_task_result(
        "t2",
        ok=True,
        elapsed_s=1.0,
        retries=0,
        input_tokens=1_000_000,
        output_tokens=0,
        model="claude-haiku-4-5-20251001",
    )
    opus_cost = tracking._task_log[0]["cost"]
    haiku_cost = tracking._task_log[1]["cost"]
    assert opus_cost > haiku_cost


def test_log_task_result_openai_model_cost():
    log_task_result(
        "design_data_schema",
        ok=True,
        elapsed_s=5.0,
        retries=0,
        input_tokens=1_000_000,
        output_tokens=0,
        model="gpt-5.3-chat-latest",
    )
    assert tracking._task_log[0]["cost"] > 0


def test_log_task_result_unknown_model_zero_cost():
    log_task_result(
        "some_task",
        ok=True,
        elapsed_s=1.0,
        retries=0,
        input_tokens=1000,
        output_tokens=500,
        model="unknown-future-model",
    )
    assert tracking._task_log[0]["cost"] == 0.0


def test_multiple_tasks_accumulate_in_log():
    for i in range(3):
        log_task_result(f"task_{i}", ok=True, elapsed_s=1.0, retries=0, model="claude-sonnet-4-6")
    assert len(tracking._task_log) == 3


# ── setup_mlflow does not call langchain autolog ──────────────────


def test_tracking_module_does_not_import_mlflow_langchain():
    src = open(tracking.__file__).read()  # noqa: SIM115
    assert "mlflow.langchain" not in src
