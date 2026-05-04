import json

import pytest

from buc_factory.agent.entity import DomainConfig
from buc_factory.agent.validator import MAX_RETRIES_PER_TASK, validate_and_extract


@pytest.fixture
def cfg() -> DomainConfig:
    return DomainConfig(
        industry="insurance",
        company_context="Test company",
        location="Paris",
        language="French",
        role="Data Analyst",
        seniority="Mid",
        tool="Power BI Desktop",
        duration_minutes=60,
        dimensions={"segment": ["A", "B"], "region": ["North", "South"]},
        entities=["policies", "clients", "claims"],
    )


# ── bootstrap_domain ──────────────────────────────────────────────


def test_bootstrap_domain_valid(tmp_path, cfg):
    data = {
        "dimensions": {"segment": ["A", "B"]},
        "entities": ["policies", "clients", "claims"],
    }
    (tmp_path / "bootstrap.json").write_text(json.dumps(data))
    ok, _, updates = validate_and_extract("bootstrap_domain", cfg, tmp_path, {})
    assert ok
    assert updates["bootstrapped_entities"] == ["policies", "clients", "claims"]
    assert updates["bootstrapped_dimensions"] == {"segment": ["A", "B"]}


def test_bootstrap_domain_missing_entities(tmp_path, cfg):
    (tmp_path / "bootstrap.json").write_text(
        json.dumps({"dimensions": {"a": ["x"]}, "entities": []})
    )
    ok, _, _ = validate_and_extract("bootstrap_domain", cfg, tmp_path, {})
    assert not ok


def test_bootstrap_domain_too_few_entities(tmp_path, cfg):
    (tmp_path / "bootstrap.json").write_text(
        json.dumps({"dimensions": {"a": ["x"]}, "entities": ["x", "y"]})
    )
    ok, msg, _ = validate_and_extract("bootstrap_domain", cfg, tmp_path, {})
    assert not ok
    assert "3" in msg


def test_bootstrap_domain_missing_file(tmp_path, cfg):
    ok, msg, _ = validate_and_extract("bootstrap_domain", cfg, tmp_path, {})
    assert not ok
    assert "validator crashed" in msg


# ── roll_scenario ─────────────────────────────────────────────────


def test_roll_scenario_valid(tmp_path, cfg):
    scenario = {"segment": "A", "region": "North"}
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    state = {"bootstrapped_dimensions": {"segment": ["A", "B"], "region": ["North", "South"]}}
    ok, _, updates = validate_and_extract("roll_scenario", cfg, tmp_path, state)
    assert ok
    assert updates["scenario"] == scenario


def test_roll_scenario_missing_dimension(tmp_path, cfg):
    (tmp_path / "scenario.json").write_text(json.dumps({"segment": "A"}))
    state = {"bootstrapped_dimensions": {"segment": ["A", "B"], "region": ["North", "South"]}}
    ok, msg, _ = validate_and_extract("roll_scenario", cfg, tmp_path, state)
    assert not ok
    assert "region" in msg


def test_roll_scenario_falls_back_to_cfg_dimensions(tmp_path, cfg):
    scenario = {"segment": "A", "region": "North"}
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    ok, _, _ = validate_and_extract("roll_scenario", cfg, tmp_path, {})
    assert ok


def test_roll_scenario_empty_dimensions_always_passes(tmp_path):
    cfg_no_dims = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Power BI Desktop",
        duration_minutes=60,
    )
    (tmp_path / "scenario.json").write_text(json.dumps({"anything": "value"}))
    ok, _, _ = validate_and_extract("roll_scenario", cfg_no_dims, tmp_path, {})
    assert ok


# ── write_brief ───────────────────────────────────────────────────


def test_write_brief_valid(tmp_path, cfg):
    (tmp_path / "brief").mkdir()
    (tmp_path / "brief" / "candidate_brief.md").write_text("x" * 1500)
    ok, _, _ = validate_and_extract("write_brief", cfg, tmp_path, {})
    assert ok


def test_write_brief_too_short(tmp_path, cfg):
    (tmp_path / "brief").mkdir()
    (tmp_path / "brief" / "candidate_brief.md").write_text("x" * 100)
    ok, msg, _ = validate_and_extract("write_brief", cfg, tmp_path, {})
    assert not ok
    assert "short" in msg


def test_write_brief_missing_file(tmp_path, cfg):
    ok, msg, _ = validate_and_extract("write_brief", cfg, tmp_path, {})
    assert not ok
    assert "not created" in msg


# ── design_data_schema ────────────────────────────────────────────


def test_design_data_schema_valid(tmp_path, cfg):
    (tmp_path / "brief").mkdir()
    schema = {
        "files": [
            {"filename": "policies.csv"},
            {"filename": "clients.csv"},
            {"filename": "claims.csv"},
        ],
        "traps": ["intentional NULL in join key"],
    }
    (tmp_path / "brief" / "data_schema.json").write_text(json.dumps(schema))
    state = {"bootstrapped_entities": ["policies", "clients", "claims"]}
    ok, _, _ = validate_and_extract("design_data_schema", cfg, tmp_path, state)
    assert ok


def test_design_data_schema_insufficient_coverage(tmp_path, cfg):
    (tmp_path / "brief").mkdir()
    schema = {
        "files": [{"filename": "policies.csv"}],
        "traps": ["a trap"],
    }
    (tmp_path / "brief" / "data_schema.json").write_text(json.dumps(schema))
    state = {"bootstrapped_entities": ["policies", "clients", "claims"]}
    ok, msg, _ = validate_and_extract("design_data_schema", cfg, tmp_path, state)
    assert not ok
    assert "entities" in msg


def test_design_data_schema_no_traps(tmp_path, cfg):
    (tmp_path / "brief").mkdir()
    schema = {
        "files": [
            {"filename": "policies.csv"},
            {"filename": "clients.csv"},
            {"filename": "claims.csv"},
        ],
        "traps": [],
    }
    (tmp_path / "brief" / "data_schema.json").write_text(json.dumps(schema))
    state = {"bootstrapped_entities": ["policies", "clients", "claims"]}
    ok, msg, _ = validate_and_extract("design_data_schema", cfg, tmp_path, state)
    assert not ok
    assert "trap" in msg


# ── generate_data_script ─────────────────────────────────────────


def test_generate_data_script_single_entity_no_pandas(tmp_path, cfg):
    csv_dir = tmp_path / "starter" / "data"
    csv_dir.mkdir(parents=True)
    (csv_dir / "policies.csv").write_text("id,name\n1,A\n")
    ok, _, _ = validate_and_extract(
        "generate_data_script", cfg, tmp_path, {"bootstrapped_entities": ["policies"]}
    )
    assert ok


def test_generate_data_script_missing_csv(tmp_path, cfg):
    (tmp_path / "starter" / "data").mkdir(parents=True)
    ok, msg, _ = validate_and_extract(
        "generate_data_script",
        cfg,
        tmp_path,
        {"bootstrapped_entities": ["policies", "clients"]},
    )
    assert not ok
    assert "policies" in msg or "clients" in msg


def test_generate_data_script_fk_violation(tmp_path, cfg):
    pytest.importorskip("pandas")
    csv_dir = tmp_path / "starter" / "data"
    csv_dir.mkdir(parents=True)
    (csv_dir / "policies.csv").write_text("id,client_id\n1,10\n2,99\n")
    (csv_dir / "clients.csv").write_text("client_id,name\n10,Alice\n11,Bob\n")
    ok, msg, _ = validate_and_extract(
        "generate_data_script",
        cfg,
        tmp_path,
        {"bootstrapped_entities": ["policies", "clients"]},
    )
    assert not ok
    assert "FK" in msg or "orphan" in msg


# ── generate_starter ─────────────────────────────────────────────


def test_generate_starter_pbip_valid(tmp_path, cfg):
    starter = tmp_path / "starter"
    for path, content in [
        ("Assessment.pbip", "{}"),
        ("Assessment.SemanticModel/definition.pbism", '{"version": "4.0", "settings": {}}'),
        ("Assessment.SemanticModel/definition/model.tmdl", "content"),
        ("Assessment.Report/definition.pbir", "{}"),
        ("Assessment.Report/report.json", "{}"),
    ]:
        p = starter / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    ok, _, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert ok


def test_generate_starter_pbip_invalid_json(tmp_path, cfg):
    starter = tmp_path / "starter"
    for path, content in [
        ("Assessment.pbip", "{bad json}"),
        ("Assessment.SemanticModel/definition.pbism", "content"),
        ("Assessment.SemanticModel/definition/model.tmdl", "content"),
        ("Assessment.Report/definition.pbir", "{}"),
        ("Assessment.Report/report.json", "{}"),
    ]:
        p = starter / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    ok, msg, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert not ok
    assert "JSON" in msg


def test_generate_starter_pbip_missing_file(tmp_path, cfg):
    (tmp_path / "starter").mkdir()
    ok, msg, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert not ok
    assert "missing" in msg


def test_generate_starter_twbx_valid(tmp_path):
    cfg = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Tableau Desktop",
        duration_minutes=60,
        deliverable_format="TWBX",
    )
    starter = tmp_path / "starter"
    starter.mkdir()
    (starter / "Assessment.tds").write_text("<datasource/>")
    (starter / "Assessment.twb").write_text("<workbook/>")
    ok, _, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert ok


def test_generate_starter_generic_enough_files(tmp_path):
    cfg = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Looker",
        duration_minutes=60,
        deliverable_format="LookML",
    )
    starter = tmp_path / "starter"
    starter.mkdir()
    (starter / "model.lkml").write_text("explore: test {}")
    (starter / "view.lkml").write_text("view: test {}")
    ok, _, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert ok


def test_generate_starter_generic_empty(tmp_path):
    cfg = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Looker",
        duration_minutes=60,
        deliverable_format="LookML",
    )
    (tmp_path / "starter").mkdir()
    ok, msg, _ = validate_and_extract("generate_starter", cfg, tmp_path, {})
    assert not ok
    assert "empty" in msg


# ── write_recruiter_solution ──────────────────────────────────────


def test_write_recruiter_solution_valid_powerbi(tmp_path, cfg):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "recruiter_solution.md").write_text("DAX\n" + "x" * 3000)
    ok, _, _ = validate_and_extract("write_recruiter_solution", cfg, tmp_path, {})
    assert ok


def test_write_recruiter_solution_too_short(tmp_path, cfg):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "recruiter_solution.md").write_text("DAX\n" + "x" * 100)
    ok, msg, _ = validate_and_extract("write_recruiter_solution", cfg, tmp_path, {})
    assert not ok
    assert "short" in msg


def test_write_recruiter_solution_missing_keyword(tmp_path, cfg):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "recruiter_solution.md").write_text("no keyword\n" + "x" * 3000)
    ok, msg, _ = validate_and_extract("write_recruiter_solution", cfg, tmp_path, {})
    assert not ok
    assert "DAX" in msg or "missing" in msg


def test_write_recruiter_solution_tableau_keyword(tmp_path):
    cfg = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Tableau Desktop",
        duration_minutes=60,
    )
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "recruiter_solution.md").write_text(
        "Tableau calculation example\n" + "x" * 3000
    )
    ok, _, _ = validate_and_extract("write_recruiter_solution", cfg, tmp_path, {})
    assert ok


def test_write_recruiter_solution_missing_file(tmp_path, cfg):
    ok, msg, _ = validate_and_extract("write_recruiter_solution", cfg, tmp_path, {})
    assert not ok
    assert "not created" in msg


# ── final_assembly ────────────────────────────────────────────────


def test_final_assembly_all_present(tmp_path, cfg):
    (tmp_path / "scenario.json").write_text("{}")
    (tmp_path / "bootstrap.json").write_text("{}")
    (tmp_path / "brief").mkdir()
    (tmp_path / "brief" / "candidate_brief.md").write_text("brief")
    (tmp_path / "brief" / "data_schema.json").write_text("{}")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "recruiter_solution.md").write_text("solution")
    (tmp_path / "starter").mkdir()
    (tmp_path / "starter" / "generate_data.py").write_text("# script")
    ok, _, _ = validate_and_extract("final_assembly", cfg, tmp_path, {})
    assert ok


def test_final_assembly_missing_artifacts(tmp_path, cfg):
    (tmp_path / "scenario.json").write_text("{}")
    ok, msg, _ = validate_and_extract("final_assembly", cfg, tmp_path, {})
    assert not ok
    assert "missing" in msg


# ── unknown task ──────────────────────────────────────────────────


def test_unknown_task_returns_false(tmp_path, cfg):
    ok, msg, _ = validate_and_extract("not_a_real_task", cfg, tmp_path, {})
    assert not ok
    assert "unknown" in msg


# ── constant ──────────────────────────────────────────────────────


def test_max_retries_is_three():
    assert MAX_RETRIES_PER_TASK == 3
