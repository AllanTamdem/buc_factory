"""Structured scoring models — shared by scorer/, api/, and app/models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ScoringCategory = Literal["methodology", "technical", "delivery"]
ToolType = Literal["powerbi", "python_da", "python_ds"]


class ScoredDimension(BaseModel):
    name: str
    category: ScoringCategory
    score: int  # 0 to max_score (integer — LLMs are far more reliable with integers)
    max_score: int
    comment: str  # one sentence in the brief's language; evidence-based, not value-based


class ScoringResult(BaseModel):
    tool_type: ToolType
    total_score: int
    max_total_score: int
    dimensions: list[ScoredDimension]
    overall_verdict: str  # one sentence: proficiency band + key strength + main gap
    language: str  # ISO 639-1 code detected from the brief

    def to_markdown(self) -> str:
        """Render as a plain markdown table (no embedded JSON)."""
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

        rows = [
            f"| {h_dim} | {h_cat} | {h_score} | {h_comment} |",
            "|---|---|---|---|",
        ]
        for d in self.dimensions:
            rows.append(
                f"| {d.name.replace('_', ' ').title()} "
                f"| {d.category} "
                f"| {d.score}/{d.max_score} "
                f"| {d.comment} |"
            )
        rows.append(
            f"| {h_total} | | **{self.total_score}/{self.max_total_score}** | "
            f"{h_verdict}: {self.overall_verdict} |"
        )
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
                v = re.search(rf"\*\*{verdict_key}\*\*\s*:\s*(.+)", comment_raw)
                overall_verdict = v.group(1).strip() if v else comment_raw.strip("* ")
                continue
            m = re.search(r"(\d+)/(\d+)", score_raw)
            score, max_score = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            cat = cat_raw.strip()
            if cat not in ("methodology", "technical", "delivery"):
                cat = "methodology"
            dimensions.append(
                ScoredDimension(
                    name=name_raw.strip("* "),
                    category=cat,  # type: ignore[arg-type]
                    score=score,
                    max_score=max_score,
                    comment=comment_raw.strip("* "),
                )
            )

        return cls(
            tool_type=tool_type,
            total_score=total_score,
            max_total_score=max_total_score,
            dimensions=dimensions,
            overall_verdict=overall_verdict,
            language=language,
        )
