"""Structured scoring models — shared by scorer/, api/, and app/models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ScoringCategory = Literal["methodology", "technical", "delivery"]
ToolType = Literal["powerbi", "python_da", "python_ds"]

# Category label translations per language
_CAT_LABELS: dict[str, dict[str, str]] = {
    "fr": {"methodology": "Méthodologie", "technical": "Technique", "delivery": "Livraison"},
    "de": {"methodology": "Methodik", "technical": "Technisch", "delivery": "Lieferung"},
    "nl": {"methodology": "Methodologie", "technical": "Technisch", "delivery": "Levering"},
    "es": {"methodology": "Metodología", "technical": "Técnico", "delivery": "Entrega"},
    "en": {"methodology": "Methodology", "technical": "Technical", "delivery": "Delivery"},
}

# Reverse map: translated category label → canonical English key
_CAT_REVERSE: dict[str, ScoringCategory] = {
    label.lower(): key  # type: ignore[misc]
    for lang_map in _CAT_LABELS.values()
    for key, label in lang_map.items()
}


class ScoredDimension(BaseModel):
    name: str
    display_name: str | None = None  # localized display name in the brief's language
    category: ScoringCategory
    score: int  # 0 to max_score (integer — LLMs are far more reliable with integers)
    max_score: int
    comment: str  # one or two sentences in the brief's language; evidence-based, not value-based


class ScoringResult(BaseModel):
    tool_type: ToolType
    total_score: int
    max_total_score: int
    dimensions: list[ScoredDimension]
    overall_verdict: str  # proficiency level + use-case strength + gap + recommendation
    language: str  # ISO 639-1 code detected from the brief

    def to_markdown(self) -> str:
        """Render as a language-aware markdown table followed by a verdict section."""
        lang = self.language.lower()
        if lang == "fr":
            h_dim, h_cat, h_score, h_comment = "Dimension", "Catégorie", "Score", "Commentaire"
            h_total, h_verdict = "**Total**", "**Verdict**"
        elif lang == "de":
            h_dim, h_cat, h_score, h_comment = "Dimension", "Kategorie", "Punkte", "Kommentar"
            h_total, h_verdict = "**Gesamt**", "**Urteil**"
        elif lang == "nl":
            h_dim, h_cat, h_score, h_comment = "Dimensie", "Categorie", "Score", "Opmerking"
            h_total, h_verdict = "**Totaal**", "**Oordeel**"
        elif lang == "es":
            h_dim, h_cat, h_score, h_comment = "Dimensión", "Categoría", "Puntuación", "Comentario"
            h_total, h_verdict = "**Total**", "**Veredicto**"
        else:
            h_dim, h_cat, h_score, h_comment = "Dimension", "Category", "Score", "Comment"
            h_total, h_verdict = "**Total**", "**Verdict**"

        cat_labels = _CAT_LABELS.get(lang, _CAT_LABELS["en"])

        rows = [
            f"| {h_dim} | {h_cat} | {h_score} | {h_comment} |",
            "|---|---|---|---|",
        ]
        for d in self.dimensions:
            display = d.display_name or d.name.replace("_", " ").title()
            cat_label = cat_labels.get(d.category, d.category)
            rows.append(f"| {display} | {cat_label} | {d.score}/{d.max_score} | {d.comment} |")
        rows.append(f"| {h_total} | | **{self.total_score}/{self.max_total_score}** | |")
        rows.append("")
        rows.append(f"{h_verdict}: {self.overall_verdict}")
        return "\n".join(rows)

    @classmethod
    def from_markdown(cls, text: str, tool_type: ToolType = "powerbi") -> ScoringResult:
        """Reconstruct from a markdown table produced by ``to_markdown()``.

        Tries the legacy embedded-JSON comment first (backward compat with old
        scoring.md files), then falls back to table parsing.
        ``tool_type`` must be supplied by the caller (derived from MLflow run params).
        """
        import json
        import re

        m = re.search(r"<!-- scoring_data: (.+?) -->", text, re.DOTALL)
        if m:
            return cls.model_validate(json.loads(m.group(1)))

        _LANG_MAP = {
            "catégorie": "fr",
            "kategorie": "de",
            "categorie": "nl",
            "categoría": "es",
        }
        _VERDICT_KEYS = {
            "fr": "Verdict",
            "de": "Urteil",
            "nl": "Oordeel",
            "es": "Veredicto",
        }

        language = "en"
        table_rows = [
            row for row in text.splitlines() if row.startswith("|") and not row.startswith("|---")
        ]
        if table_rows:
            header_cat = table_rows[0].split("|")[2].strip().lower()
            language = _LANG_MAP.get(header_cat, "en")

        verdict_key = _VERDICT_KEYS.get(language, "Verdict")
        dimensions: list[ScoredDimension] = []
        total_score = max_total_score = 0
        overall_verdict = ""

        for row in table_rows[1:]:  # skip header row
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) < 4:
                continue
            name_raw, cat_raw, score_raw, comment_raw = (cells[0], cells[1], cells[2], cells[3])
            if "**" in name_raw:  # total row
                m = re.search(r"\*\*(\d+)/(\d+)\*\*", score_raw)
                if m:
                    total_score, max_total_score = int(m.group(1)), int(m.group(2))
                # Legacy format: verdict was in comment cell of total row
                v = re.search(rf"\*\*{verdict_key}\*\*\s*:\s*(.+)", comment_raw)
                overall_verdict = v.group(1).strip() if v else ""
                continue
            m = re.search(r"(\d+)/(\d+)", score_raw)
            score, max_score = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            cat = _CAT_REVERSE.get(cat_raw.strip().lower(), "methodology")
            dimensions.append(
                ScoredDimension(
                    name=name_raw.strip("* "),
                    category=cat,
                    score=score,
                    max_score=max_score,
                    comment=comment_raw.strip("* "),
                )
            )

        # New format: verdict appears as a line after the table
        if not overall_verdict:
            v = re.search(rf"\*\*{verdict_key}\*\*\s*:\s*(.+)", text)
            if v:
                overall_verdict = v.group(1).strip()

        return cls(
            tool_type=tool_type,
            total_score=total_score,
            max_total_score=max_total_score,
            dimensions=dimensions,
            overall_verdict=overall_verdict,
            language=language,
        )
