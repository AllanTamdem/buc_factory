import pytest
import yaml

from buc_factory.agent.entity import PLAN, DomainConfig


def test_plan_has_eight_tasks():
    assert len(PLAN) == 8


def test_plan_order():
    assert PLAN[0] == "bootstrap_domain"
    assert PLAN[-1] == "final_assembly"


def test_plan_contains_required_tasks():
    for task in ("roll_scenario", "write_brief", "generate_data_script", "write_recruiter_solution"):
        assert task in PLAN


def test_domain_config_required_fields():
    cfg = DomainConfig(
        industry="banking",
        company_context="A retail bank",
        location="London",
        language="English",
        role="Analyst",
        seniority="Senior",
        tool="Tableau",
        duration_minutes=90,
    )
    assert cfg.industry == "banking"
    assert cfg.tool == "Tableau"


def test_domain_config_defaults():
    cfg = DomainConfig(
        industry="test",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Power BI Desktop",
        duration_minutes=60,
    )
    assert cfg.dimensions is None
    assert cfg.entities is None
    assert cfg.deliverable_format == "PBIP"


def test_domain_config_from_yaml(tmp_path):
    data = {
        "industry": "retail",
        "company_context": "a retailer",
        "location": "Berlin",
        "language": "German",
        "role": "Analyst",
        "seniority": "Junior",
        "tool": "Looker",
        "duration_minutes": 45,
        "deliverable_format": "LookML",
        "entities": ["sales", "products", "stores"],
    }
    cfg_file = tmp_path / "cfg.yml"
    cfg_file.write_text(yaml.dump(data))
    cfg = DomainConfig.from_yaml(cfg_file)
    assert cfg.industry == "retail"
    assert cfg.deliverable_format == "LookML"
    assert cfg.entities == ["sales", "products", "stores"]


def test_domain_config_from_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DomainConfig.from_yaml(tmp_path / "missing.yml")
