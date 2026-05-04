import pytest

from buc_factory.agent.entity import PLAN
from buc_factory.agent.graph import (
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MODEL,
    _LARGE_MAX_TOKENS,
    _PARALLEL_GROUP_NEXT,
    _PARALLEL_GROUPS,
    _TASKS_WITHOUT_READ_FILE,
    _task_config,
)

# ── _TASKS_WITHOUT_READ_FILE ──────────────────────────────────────


def test_tasks_without_read_file_exact_members():
    assert {
        "bootstrap_domain",
        "roll_scenario",
        "write_brief",
        "design_data_schema",
    } == _TASKS_WITHOUT_READ_FILE


def test_tasks_with_valid_read_file_use_not_restricted():
    for task in (
        "generate_data_script",
        "generate_starter",
        "write_recruiter_solution",
        "final_assembly",
    ):
        assert task not in _TASKS_WITHOUT_READ_FILE


def test_all_restricted_tasks_exist_in_plan():
    for task in _TASKS_WITHOUT_READ_FILE:
        assert task in PLAN


# ── _task_config ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task", ["write_recruiter_solution", "generate_data_script", "generate_starter"]
)
def test_large_tasks_get_large_token_budget(task):
    max_tok, _ = _task_config(task)
    assert max_tok == _LARGE_MAX_TOKENS


@pytest.mark.parametrize(
    "task",
    ["bootstrap_domain", "roll_scenario", "write_brief", "design_data_schema", "final_assembly"],
)
def test_standard_tasks_get_default_token_budget(task):
    max_tok, _ = _task_config(task)
    assert max_tok == _DEFAULT_MAX_TOKENS


def test_haiku_tasks_use_haiku_model():
    for task in ("roll_scenario", "final_assembly"):
        _, model = _task_config(task)
        assert "haiku" in model


def test_opus_tasks_use_opus_model():
    for task in ("generate_starter", "write_recruiter_solution"):
        _, model = _task_config(task)
        assert "opus" in model


def test_default_tasks_use_sonnet_model():
    for task in ("bootstrap_domain", "write_brief", "design_data_schema", "generate_data_script"):
        _, model = _task_config(task)
        assert model == _DEFAULT_MODEL


# ── _PARALLEL_GROUPS ──────────────────────────────────────────────


def test_parallel_group_keys_are_valid_plan_indices():
    for idx in _PARALLEL_GROUPS:
        assert 0 <= idx < len(PLAN)


def test_parallel_group_tasks_exist_in_plan():
    for tasks in _PARALLEL_GROUPS.values():
        for task in tasks:
            assert task in PLAN


def test_parallel_group_next_matches_derived_formula():
    for group_idx, next_idx in _PARALLEL_GROUP_NEXT.items():
        expected = max(PLAN.index(n) for n in _PARALLEL_GROUPS[group_idx]) + 1
        assert next_idx == expected


def test_parallel_group_next_is_within_plan():
    for next_idx in _PARALLEL_GROUP_NEXT.values():
        assert next_idx <= len(PLAN)


def test_parallel_groups_advance_to_correct_tasks():
    # group at idx 2 (write_brief, design_data_schema) → next is generate_data_script
    assert PLAN[_PARALLEL_GROUP_NEXT[2]] == "generate_data_script"
    # group at idx 5 (generate_starter, write_recruiter_solution) → next is final_assembly
    assert PLAN[_PARALLEL_GROUP_NEXT[5]] == "final_assembly"
