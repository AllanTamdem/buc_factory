import pytest

from buc_factory.agent.entity import DomainConfig


@pytest.fixture
def cfg() -> DomainConfig:
    return DomainConfig(
        industry="insurance",
        company_context="A mid-sized insurance company",
        location="Paris, France",
        language="French",
        role="Data Analyst",
        seniority="Mid",
        tool="Power BI Desktop",
        duration_minutes=60,
        dimensions={"segment": ["A", "B"], "region": ["North", "South"]},
        entities=["policies", "clients", "claims"],
    )
