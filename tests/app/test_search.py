"""Tests for app/search.py — build_index_text and search helpers."""

from buc_factory.app.search import build_index_text

_BASE_PARAMS = {
    "industry": "assurance vie",
    "role": "Data Scientist",
    "tool": "Python",
    "language": "French",
    "seniority": "Senior",
    "location": "Paris",
}


# ── build_index_text ──────────────────────────────────────────────


def test_build_index_text_contains_labeled_fields():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert "Role: Data Scientist" in text
    assert "Industry: assurance vie" in text
    assert "Tool: Python" in text
    assert "Language: French" in text
    assert "Seniority: Senior" in text
    assert "Location: Paris" in text


def test_build_index_text_no_scenario_no_scenario_section():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert "Scenario:" not in text


def test_build_index_text_with_scalar_scenario_values():
    scenario = {"angle": "churn analysis", "volume": "500k"}
    text = build_index_text(_BASE_PARAMS, scenario=scenario)
    assert "Scenario:" in text
    assert "churn analysis" in text
    assert "500k" in text


def test_build_index_text_with_list_scenario_values():
    scenario = {"gamme": ["Fonds euros", "UC", "PER"]}
    text = build_index_text(_BASE_PARAMS, scenario=scenario)
    assert "Fonds euros" in text
    assert "UC" in text
    assert "PER" in text
    # Must NOT contain the raw list repr
    assert "['" not in text


def test_build_index_text_missing_params_uses_empty_string():
    text = build_index_text({}, scenario=None)
    # All labels present but values empty — no crash
    assert "Role:" in text
    assert "Industry:" in text


def test_build_index_text_role_comes_first():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert text.startswith("Role:")


def test_build_index_text_data_scientist_vs_analyst_are_distinct():
    scientist = build_index_text({**_BASE_PARAMS, "role": "Data Scientist"}, scenario=None)
    analyst = build_index_text({**_BASE_PARAMS, "role": "Data Analyst"}, scenario=None)
    assert scientist != analyst
    assert "Data Scientist" in scientist
    assert "Data Analyst" in analyst
