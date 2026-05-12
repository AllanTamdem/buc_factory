import pytest

from buc_factory.agent.entity import DomainConfig
from buc_factory.agent.prompting import _is_python, calc_language, raw_system_template, task_prompt

_NA = "(not yet available)"


@pytest.fixture
def cfg_python_ds() -> DomainConfig:
    return DomainConfig(
        industry="retail",
        company_context="A retailer",
        location="London",
        language="English",
        role="Data Scientist",
        seniority="Mid",
        tool="Python (Jupyter)",
        duration_minutes=90,
        deliverable_format="IPYNB",
    )


@pytest.fixture
def cfg_python_da() -> DomainConfig:
    return DomainConfig(
        industry="saas",
        company_context="A SaaS company",
        location="Paris",
        language="French",
        role="Data Analyst",
        seniority="Junior",
        tool="Python",
        duration_minutes=60,
        deliverable_format="IPYNB",
    )


# ── fallback "(not yet available)" ────────────────────────────────


@pytest.mark.parametrize(
    "task_name,missing_kwarg",
    [
        ("write_brief", "scenario_json"),
        ("write_brief", "bootstrap_json"),
        ("design_data_schema", "bootstrap_json"),
        ("generate_data_script", "data_schema_json"),
        ("generate_data_script", "candidate_brief"),
        ("generate_starter", "data_schema_json"),
        ("write_recruiter_solution", "scenario_json"),
        ("write_recruiter_solution", "data_schema_json"),
        ("write_recruiter_solution", "candidate_brief"),
    ],
)
def test_task_prompt_fallback_when_kwarg_is_none(cfg, task_name, missing_kwarg):
    prompt = task_prompt(task_name, cfg, None)
    assert _NA in prompt, f"{task_name} should contain fallback for missing {missing_kwarg}"


# ── pre-loaded content is embedded ────────────────────────────────


def test_write_brief_embeds_scenario_and_bootstrap(cfg):
    p = task_prompt(
        "write_brief",
        cfg,
        None,
        scenario_json='{"region": "North"}',
        bootstrap_json='{"entities": ["claims"]}',
    )
    assert '{"region": "North"}' in p
    assert '{"entities": ["claims"]}' in p
    assert _NA not in p


def test_design_data_schema_embeds_bootstrap(cfg):
    p = task_prompt("design_data_schema", cfg, None, bootstrap_json='{"entities": ["orders"]}')
    assert '{"entities": ["orders"]}' in p
    assert _NA not in p


def test_generate_data_script_embeds_schema_and_brief(cfg):
    p = task_prompt(
        "generate_data_script",
        cfg,
        None,
        data_schema_json='{"files": []}',
        candidate_brief="# Brief\n\nQ1: something",
    )
    assert '{"files": []}' in p
    assert "# Brief" in p
    assert _NA not in p


def test_generate_starter_powerbi_embeds_schema(cfg):
    p = task_prompt(
        "generate_starter", cfg, None, data_schema_json='{"files": [{"filename": "policies.csv"}]}'
    )
    assert '"filename": "policies.csv"' in p
    assert _NA not in p


def test_generate_starter_python_ds_embeds_schema(cfg_python_ds):
    p = task_prompt("generate_starter", cfg_python_ds, None, data_schema_json='{"files": []}')
    assert '{"files": []}' in p
    assert "notebook.ipynb" in p
    assert "requirements.txt" in p
    assert "xgboost" in p
    assert "tensorflow" in p
    assert "Modelling" in p


def test_generate_starter_python_da_embeds_schema(cfg_python_da):
    p = task_prompt("generate_starter", cfg_python_da, None, data_schema_json='{"files": []}')
    assert '{"files": []}' in p
    assert "notebook.ipynb" in p
    assert "requirements.txt" in p
    assert "xgboost" not in p
    assert "tensorflow" not in p


def test_write_recruiter_solution_embeds_all_three(cfg):
    p = task_prompt(
        "write_recruiter_solution",
        cfg,
        None,
        scenario_json='{"segment": "A"}',
        data_schema_json='{"files": []}',
        candidate_brief="# Brief\n\nBuild a dashboard.",
    )
    assert '{"segment": "A"}' in p
    assert '{"files": []}' in p
    assert "Build a dashboard." in p
    assert _NA not in p


# ── retry suffix is appended ──────────────────────────────────────


def test_retry_suffix_appended(cfg):
    p = task_prompt("write_brief", cfg, "Output was too short")
    assert "RETRY FEEDBACK" in p
    assert "Output was too short" in p


def test_no_retry_suffix_when_feedback_is_none(cfg):
    p = task_prompt("write_brief", cfg, None)
    assert "RETRY FEEDBACK" not in p


# ── tasks without new kwargs still render ─────────────────────────


def test_bootstrap_domain_renders(cfg):
    p = task_prompt("bootstrap_domain", cfg, None)
    assert "bootstrap" in p.lower()


def test_roll_scenario_renders_with_rolled(cfg):
    p = task_prompt("roll_scenario", cfg, None, rolled={"segment": "A", "region": "North"})
    assert '"segment": "A"' in p


def test_final_assembly_renders(cfg):
    p = task_prompt("final_assembly", cfg, None)
    assert "final" in p.lower() or "assembly" in p.lower()


# ── calc_language helper ────────────────────────────────────────────


def test_calc_language_power_bi():
    assert calc_language("Power BI Desktop") == "DAX"


def test_calc_language_python():
    assert "Python" in calc_language("Python (Jupyter)")
    assert "pandas" in calc_language("Python")
    assert "Python" in calc_language("Jupyter Notebook")


def test_calc_language_unknown():
    assert calc_language("Some BI Tool") == "calculation"


def test_is_python():
    assert _is_python("Python")
    assert _is_python("Python (Jupyter)")
    assert _is_python("Jupyter Notebook")
    assert not _is_python("Power BI")
    assert not _is_python("Tableau Desktop")


# ── raw_system_template ───────────────────────────────────────────


def test_raw_system_template_returns_string():
    tmpl = raw_system_template("claude")
    assert isinstance(tmpl, str)
    assert len(tmpl) > 100


def test_raw_system_template_contains_placeholders():
    tmpl = raw_system_template("claude")
    # Template must not have been rendered — cfg variables must still be present
    assert "{" in tmpl


def test_raw_system_template_does_not_contain_cfg_values(cfg):
    tmpl = raw_system_template("claude")
    assert cfg.industry not in tmpl
    assert cfg.role not in tmpl


def test_raw_system_template_openai_returns_string():
    tmpl = raw_system_template("openai")
    assert isinstance(tmpl, str)
    assert len(tmpl) > 100


def test_raw_system_template_providers_may_differ():
    claude_tmpl = raw_system_template("claude")
    openai_tmpl = raw_system_template("openai")
    # They may be equal (if no patch file) or different — both are valid
    assert isinstance(claude_tmpl, str)
    assert isinstance(openai_tmpl, str)
