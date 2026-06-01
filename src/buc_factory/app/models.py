"""Pydantic input and output models for the BUC Factory API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Dimensions(BaseModel):
    """Dimension name → list of options. Any field is optional; keys are industry-specific."""

    model_config = ConfigDict(extra="allow")


# ── inputs ─────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "industry": "assurance vie",
                "company_context": (
                    "Un assureur vie français de taille intermédiaire, distribué via un réseau"
                    " de conseillers en gestion de patrimoine et des partenariats bancaires."
                    " Gamme couvrant les contrats monosupport euros, multisupports UC et PER."
                ),
                "location": "Paris, France",
                "language": "French",
                "role": "Data Analyst",
                "seniority": "Mid-Senior",
                "tool": "Power BI Desktop",
                "duration_minutes": 75,
                "deliverable_format": "PBIP",
                "dimensions": {
                    "gamme": [
                        "Fonds euros garanti",
                        "Unités de compte (UC)",
                        "Contrat multisupport euros / UC",
                        "Plan d'Épargne Retraite (PER)",
                    ],
                    "angle": [
                        "Performance de l'épargne (rendements, arbitrages UC/euros)",
                        "Comportement de rachat (total vs partiel, taux de fuite)",
                        "Collecte nette (versements – rachats – prestations décès)",
                        "Analyse de la mortalité et provisions mathématiques",
                    ],
                    "historique_mois": ["12", "24", "36", "60"],
                    "volumetrie": ["30k_contrats", "100k_contrats", "500k_contrats"],
                    "twist": [
                        "Hausse des taux : arbitrages massifs UC → fonds euros",
                        "Réforme PER : migration des anciens contrats Madelin",
                        "Nouvelle réglementation DDA sur les exigences de conseil",
                        "Lancement d'un fonds euros boosté il y a 6 mois",
                    ],
                    "restitution": [
                        "Tableau de bord Direction Technique / Actuariat",
                        "Reporting commercial réseau de distribution",
                        "Dashboard risque de rachat (direction ALM)",
                    ],
                },
                "entities": [
                    "contrats",
                    "clients",
                    "versements",
                    "rachats",
                    "produits",
                    "provisions",
                ],
            }
        }
    )

    industry: str
    company_context: str
    location: str
    language: str
    role: str
    seniority: str
    tool: str
    duration_minutes: int
    dimensions: Dimensions | None = None
    entities: list[str] | None = None
    deliverable_format: str = "PBIP"


# ── outputs ────────────────────────────────────────────────────────


class RunParameters(BaseModel):
    industry: str | None = None
    role: str | None = None
    seniority: str | None = None
    tool: str | None = None
    language: str | None = None
    location: str | None = None
    duration_minutes: int | None = None
    deliverable_format: str | None = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    mlflow_run_id: str | None = None
    parameters: RunParameters | None = None
    scenario: dict[str, Any] | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]


class RunResponse(BaseModel):
    run_id: str
    status: str


class SearchResult(BaseModel):
    score: float
    run_id: str
    status: str
    mlflow_run_id: str | None = None
    parameters: RunParameters | None = None
    scenario: dict[str, Any] | None = None


# ── simulation models ──────────────────────────────────────────────


class SimulationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "run_001",
                "proficiency": 0.65,
            }
        }
    )

    run_id: str = Field(description="Source assessment run to simulate against")
    proficiency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Candidate proficiency 0.0–1.0. "
            "1.0 = perfect expert submission. "
            "Sampled uniformly in [0.3, 0.95] when omitted."
        ),
    )
    alea: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Daily energy 0.0–1.0. Drawn randomly when omitted.",
    )


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str


class SimulationDetails(BaseModel):
    simulation_id: str
    source_run_id: str
    status: str
    proficiency: float | None
    alea: float | None = None
    mlflow_run_id: str | None = None
