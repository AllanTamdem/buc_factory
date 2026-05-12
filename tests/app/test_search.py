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


def test_build_index_text_contains_all_field_values():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert "Data Scientist" in text
    assert "assurance vie" in text
    assert "Python" in text
    assert "French" in text
    assert "Senior" in text
    assert "Paris" in text


def test_build_index_text_no_scenario_no_scenario_section():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert "Scenario:" not in text


def test_build_index_text_with_scalar_scenario_values():
    scenario = {"angle": "churn analysis", "volume": "500k"}
    text = build_index_text(_BASE_PARAMS, scenario=scenario)
    assert "Scenario:" in text
    assert "churn analysis" in text
    assert "500k" in text


def test_build_index_text_scenario_keys_included():
    scenario = {"angle": "churn analysis", "volume": "500k"}
    text = build_index_text(_BASE_PARAMS, scenario=scenario)
    assert "angle: churn analysis" in text
    assert "volume: 500k" in text


def test_build_index_text_with_list_scenario_values():
    scenario = {"gamme": ["Fonds euros", "UC", "PER"]}
    text = build_index_text(_BASE_PARAMS, scenario=scenario)
    assert "Fonds euros" in text
    assert "UC" in text
    assert "PER" in text
    # Must NOT contain the raw list repr
    assert "['" not in text


def test_build_index_text_missing_params_no_crash():
    text = build_index_text({}, scenario=None)
    assert isinstance(text, str)
    assert len(text) > 0


def test_build_index_text_missing_params_omits_empty_fields():
    text = build_index_text({}, scenario=None)
    assert "working in the  industry" not in text
    assert "using ," not in text
    assert "based in ," not in text


def test_build_index_text_starts_with_prose():
    text = build_index_text(_BASE_PARAMS, scenario=None)
    assert text.startswith("A ")


def test_build_index_text_data_scientist_vs_analyst_are_distinct():
    scientist = build_index_text({**_BASE_PARAMS, "role": "Data Scientist"}, scenario=None)
    analyst = build_index_text({**_BASE_PARAMS, "role": "Data Analyst"}, scenario=None)
    assert scientist != analyst
    assert "Data Scientist" in scientist
    assert "Data Analyst" in analyst
