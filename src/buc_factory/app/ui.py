"""Streamlit UI for BUC Factory — run with: streamlit run src/buc_factory/app/ui.py"""

import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from buc_factory.scorer.models import ScoringResult

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_APP_ROOT = next(
    (p for p in [Path("/app"), _HERE.parent.parent.parent] if (p / "README.md").exists()),
    _HERE.parent.parent.parent,
)
_LOGO_LOCKUP = _APP_ROOT / "docs" / "logo_lockup.svg"
_LOGO_MARK = _HERE.parent / "static" / "logo_mark.svg"
_README = _APP_ROOT / "README.md"

# ── Constants ─────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "done": "#1D9E75",
    "running": "#534AB7",
    "queued": "#F59E0B",
    "failed": "#EF4444",
    "unknown": "#9CA3AF",
}
STATUS_ICONS = {"done": "✅", "running": "⚙️", "queued": "⏳", "failed": "❌", "unknown": "❓"}

SENIORITY_OPTS = ["Junior", "Mid-Senior", "Senior", "Staff"]
TOOL_OPTS = ["Power BI Desktop", "Python (Notebook)"]
FORMAT_OPTS = ["PBIP", "IPYNB"]

SAMPLE_DEFAULTS: dict[str, Any] = {
    "industry": "P&C insurance",
    "company_context": (
        "A mid-sized Paris-based property & casualty insurer serving French retail "
        "customers across home, auto, and affinity products. ~500 employees, "
        "distributing through tied agents and direct online channels."
    ),
    "location": "Paris, France",
    "language": "French",
    "role": "Data Analyst",
    "seniority": "Mid-Senior",
    "tool": "Power BI Desktop",
    "duration_minutes": 75,
    "deliverable_format": "PBIP",
    "dimensions": "",
    "entities": "",
}

# Independent fields — randomized freely
RANDOMIZE_POOL: dict[str, Any] = {
    "role": [
        "Data Analyst",
        "Data Scientist",
        "BI Developer",
        "Analytics Engineer",
        "Product Analyst",
    ],
    "seniority": SENIORITY_OPTS,
    "tool": ["Power BI Desktop", "Python (Notebook)"],
    "duration_minutes": [60, 75, 90, 105, 120],
    "deliverable_format": FORMAT_OPTS,
}

# Correlated fields — always picked as a coherent unit
_RAND_PROFILES: list[dict[str, Any]] = [
    # ── France / French ───────────────────────────────────────────────────────
    {
        "location": "Paris, France",
        "language": "French",
        "industry": "P&C insurance",
        "company_context": (
            "A mid-sized Paris-based property & casualty insurer serving French retail "
            "customers across home, auto, and affinity products. ~500 employees, "
            "distributing through tied agents and direct online channels."
        ),
    },
    {
        "location": "Paris, France",
        "language": "French",
        "industry": "banque de détail",
        "company_context": (
            "A French retail bank with 600+ branches and a growing digital offering. "
            "Serves retail and professional clients across savings, mortgage, "
            "consumer credit, and insurance."
        ),
    },
    {
        "location": "Paris, France",
        "language": "French",
        "industry": "SaaS",
        "company_context": (
            "A B2B SaaS company headquartered in Paris, offering a project management "
            "platform to European SMEs. ~200 employees, ARR €12M, Series B. "
            "Growing 40% YoY with a strong self-serve funnel."
        ),
    },
    {
        "location": "Paris, France",
        "language": "French",
        "industry": "Telecoms",
        "company_context": (
            "A mid-market French telecom operator serving 2.5M subscribers with "
            "mobile, broadband, and IPTV bundles. Digital-first strategy, NPS 48."
        ),
    },
    {
        "location": "Lyon, France",
        "language": "French",
        "industry": "Healthcare",
        "company_context": (
            "A French private hospital group operating 12 clinics across the Auvergne-"
            "Rhône-Alpes region. ~2,000 employees, revenue €180M. Mix of ambulatory "
            "and inpatient care."
        ),
    },
    {
        "location": "Bordeaux, France",
        "language": "French",
        "industry": "E-commerce",
        "company_context": (
            "A French direct-to-consumer wine and spirits e-retailer. ~120 employees, "
            "GMV €45M. Operates across France, Belgium, and Switzerland."
        ),
    },
    {
        "location": "Brussels, Belgium",
        "language": "French",
        "industry": "Asset management",
        "company_context": (
            "A Brussels-based asset manager with €4B AUM across European equity "
            "and fixed-income strategies. 60 employees, regulated under UCITS."
        ),
    },
    # ── UK / English ──────────────────────────────────────────────────────────
    {
        "location": "London, United Kingdom",
        "language": "English",
        "industry": "Asset management",
        "company_context": (
            "A London-based asset manager with £8B AUM across equity, fixed income, "
            "and alternative strategies. 80 employees, FCA regulated."
        ),
    },
    {
        "location": "London, United Kingdom",
        "language": "English",
        "industry": "Retail banking",
        "company_context": (
            "A mid-tier UK retail bank with 400 branches and 3M current account holders. "
            "Growing challenger bank division with 600k digital-only customers."
        ),
    },
    {
        "location": "London, United Kingdom",
        "language": "English",
        "industry": "P&C insurance",
        "company_context": (
            "A specialist London market insurer writing marine, aviation, and property "
            "risks globally. ~300 employees, Lloyd's syndicate, GWP £420M."
        ),
    },
    # ── USA / English ─────────────────────────────────────────────────────────
    {
        "location": "New York, USA",
        "language": "English",
        "industry": "Retail",
        "company_context": (
            "A US omnichannel retailer with 500+ stores and a mature e-commerce channel. "
            "~15,000 employees. Focused on apparel and home goods, 35% of sales online."
        ),
    },
    {
        "location": "New York, USA",
        "language": "English",
        "industry": "Asset management",
        "company_context": (
            "A New York-based quantitative asset manager with $12B AUM. "
            "35 employees, systematic equity and macro strategies, SEC registered."
        ),
    },
    {
        "location": "Chicago, USA",
        "language": "English",
        "industry": "Healthcare",
        "company_context": (
            "A US regional health system operating 8 hospitals and 60 outpatient clinics "
            "across the Midwest. ~9,000 employees, revenue $1.4B."
        ),
    },
    # ── Germany / German ──────────────────────────────────────────────────────
    {
        "location": "Berlin, Germany",
        "language": "German",
        "industry": "SaaS",
        "company_context": (
            "A Berlin-based HR-tech SaaS company serving mid-market enterprises "
            "across the DACH region. ~150 employees, ARR €8M, Series A."
        ),
    },
    {
        "location": "Munich, Germany",
        "language": "German",
        "industry": "Manufacturing",
        "company_context": (
            "A Bavarian precision manufacturer supplying components to automotive OEMs. "
            "~1,200 employees across 3 plants. Revenue €340M, ISO 9001 certified."
        ),
    },
    {
        "location": "Zürich, Switzerland",
        "language": "German",
        "industry": "Asset management",
        "company_context": (
            "A Swiss private bank and asset manager with CHF 6B AUM. "
            "FINMA regulated, serving UHNWI clients across Europe and the Middle East."
        ),
    },
    # ── Netherlands / Dutch ───────────────────────────────────────────────────
    {
        "location": "Amsterdam, Netherlands",
        "language": "Dutch",
        "industry": "E-commerce",
        "company_context": (
            "A Dutch online marketplace for home & garden, operating across Benelux. "
            "~300 employees, GMV €280M, 1.2M active buyers."
        ),
    },
    {
        "location": "Amsterdam, Netherlands",
        "language": "Dutch",
        "industry": "Retail banking",
        "company_context": (
            "A Dutch retail bank with 2.5M customers and a predominantly digital "
            "distribution model. Regulated by DNB, active in mortgages and savings."
        ),
    },
    # ── Spain / Spanish ───────────────────────────────────────────────────────
    {
        "location": "Madrid, Spain",
        "language": "Spanish",
        "industry": "Telecoms",
        "company_context": (
            "A Spanish mobile virtual network operator serving 1.8M subscribers. "
            "~200 employees, focuses on prepaid and youth segments."
        ),
    },
    {
        "location": "Barcelona, Spain",
        "language": "Spanish",
        "industry": "E-commerce",
        "company_context": (
            "A Barcelona-based fashion e-commerce platform operating across Spain, "
            "Portugal, and Latin America. ~400 employees, GMV €120M."
        ),
    },
]

_DIM_PC = json.dumps(
    {
        "branche": [
            "MRH (Multirisque Habitation)",
            "Auto particuliers",
            "Mixte MRH + Auto",
            "Garanties affinitaires",
        ],
        "angle": [
            "Pilotage de la sinistralité (S/P, fréquence, coût moyen)",
            "Performance commerciale (acquisition, churn, transformation)",
            "Rentabilité par segment (combined ratio)",
            "Détection d'anomalies / qualité de portefeuille",
        ],
        "historique_mois": ["24", "30", "36", "48"],
        "volumetrie": ["50k_polices", "120k_polices", "250k_polices"],
    },
    indent=2,
    ensure_ascii=False,
)

_DIM_BANK = json.dumps(
    {
        "domaine": [
            "Crédit immobilier (origination & encours)",
            "Épargne (livrets, assurance-vie, dépôts à terme)",
            "Comptes courants & moyens de paiement",
        ],
        "angle": [
            "Performance commerciale (conquête, équipement, attrition)",
            "Qualité du portefeuille de crédit (taux d'impayés, NPL)",
            "Rentabilité client (PNB, coût du risque, ROE par segment)",
        ],
    },
    indent=2,
    ensure_ascii=False,
)

INDUSTRY_TEMPLATES: dict[str, Any] = {
    "P&C Insurance (France)": {
        "industry": "P&C insurance",
        "company_context": (
            "A mid-sized Paris-based property & casualty insurer serving French retail "
            "customers across home, auto, and affinity products. ~500 employees, "
            "distributing through tied agents and direct online channels."
        ),
        "location": "Paris, France",
        "language": "French",
        "role": "Data Analyst",
        "seniority": "Mid-Senior",
        "tool": "Power BI Desktop",
        "duration_minutes": 75,
        "deliverable_format": "PBIP",
        "dimensions": _DIM_PC,
        "entities": "polices\nsinistres\nclients\nproduits\ngeographie",
    },
    "Retail Banking (France)": {
        "industry": "banque de détail",
        "company_context": (
            "Une banque de réseau française de taille intermédiaire, implantée sur "
            "l'ensemble du territoire national avec un réseau d'environ 600 agences "
            "et une offre digitale en croissance."
        ),
        "location": "Paris, France",
        "language": "French",
        "role": "Data Analyst",
        "seniority": "Mid-Senior",
        "tool": "Power BI Desktop",
        "duration_minutes": 90,
        "deliverable_format": "PBIP",
        "dimensions": _DIM_BANK,
        "entities": "clients\ncomptes\ncontrats_credit\ntransactions\nagences",
    },
    "Retail Banking — Data Science (France)": {
        "industry": "banque de détail",
        "company_context": (
            "Une banque de réseau française de taille intermédiaire. Clientèle composée "
            "de particuliers et professionnels. Gamme couvrant comptes courants, épargne, "
            "crédit immobilier, crédit à la consommation."
        ),
        "location": "Paris, France",
        "language": "French",
        "role": "Data Scientist",
        "seniority": "Senior",
        "tool": "Python (Notebook)",
        "duration_minutes": 90,
        "deliverable_format": "IPYNB",
        "dimensions": "",
        "entities": "clients\ncomptes\ncontrats_credit\ntransactions",
    },
    "SaaS (France)": {
        "industry": "SaaS",
        "company_context": (
            "A B2B SaaS company headquartered in Paris, offering a project management "
            "platform to European SMEs. ~200 employees, ARR €12M, Series B. Growing 40% YoY."
        ),
        "location": "Paris, France",
        "language": "French",
        "role": "Data Analyst",
        "seniority": "Mid-Senior",
        "tool": "Power BI Desktop",
        "duration_minutes": 90,
        "deliverable_format": "PBIP",
        "dimensions": "",
        "entities": "",
    },
    "Retail (USA)": {
        "industry": "Retail",
        "company_context": (
            "A US omnichannel retailer with 500+ stores and a mature e-commerce channel. "
            "~15,000 employees. Focused on apparel and home goods."
        ),
        "location": "New York, USA",
        "language": "English",
        "role": "Data Analyst",
        "seniority": "Senior",
        "tool": "Python (Notebook)",
        "duration_minutes": 90,
        "deliverable_format": "IPYNB",
        "dimensions": "",
        "entities": "",
    },
}

# ── API helpers ───────────────────────────────────────────────────────────────


def _base() -> str:
    # Prefer the internal env URL (set in Docker for container-to-container calls).
    # Fall back to the sidebar override for local dev where API_BASE_URL is not set.
    env_url = os.environ.get("API_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return str(st.session_state.get("api_base", "http://localhost:8010")).rstrip("/")


def _public_base() -> str:
    """URL the *browser* uses to reach the API — may differ from _base() inside Docker."""
    return os.environ.get("API_PUBLIC_URL", _base()).rstrip("/")


def _api_get(path: str, params: dict[str, Any] | None = None) -> tuple[Any, str | None]:
    try:
        r = requests.get(f"{_base()}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Is the server running?"
    except requests.exceptions.HTTPError as e:
        resp = e.response
        code = resp.status_code if resp is not None else "?"
        text = resp.text if resp is not None else ""
        return None, f"HTTP {code}: {text}"
    except Exception as e:
        return None, str(e)


def _api_post(path: str, payload: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        r = requests.post(f"{_base()}{path}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Is the server running?"
    except requests.exceptions.HTTPError as e:
        resp = e.response
        code = resp.status_code if resp is not None else "?"
        text = resp.text if resp is not None else ""
        return None, f"HTTP {code}: {text}"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_text(base_url: str, path: str) -> tuple[str | None, str | None]:
    """Fetch plain-text content from the API (cached 5 min)."""
    try:
        r = requests.get(f"{base_url}{path}", timeout=20)
        r.raise_for_status()
        return r.text, None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API."
    except requests.exceptions.HTTPError as e:
        resp = e.response
        code = resp.status_code if resp is not None else "?"
        text = resp.text if resp is not None else ""
        return None, f"HTTP {code}: {text}"
    except Exception as e:
        return None, str(e)


# ── UI helpers ────────────────────────────────────────────────────────────────


def _badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#9CA3AF")
    icon = STATUS_ICONS.get(status, "❓")
    return (
        f'<span style="background:{color};color:#fff;padding:3px 11px;'
        f'border-radius:12px;font-size:12px;font-weight:600;white-space:nowrap">'
        f"{icon} {status}</span>"
    )


def _param(label: str, value: str) -> str:
    return f"<span style='font-size:11px;opacity:0.55'>{label}</span><br><b>{value}</b>"


def _section(title: str) -> None:
    st.markdown(
        f'<p style="font-size:22px;font-weight:700;'
        f'border-bottom:3px solid #534AB7;padding-bottom:8px;margin-bottom:18px">'
        f"{title}</p>",
        unsafe_allow_html=True,
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stAppDeployButton"] { display: none; }

        [data-testid="stHeader"] {
            position: relative;
        }
        [data-testid="stHeader"]::after {
            content: "© 2026 Alain TAMDEM";
            position: absolute;
            right: 24px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
            opacity: 0.55;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            letter-spacing: 0.3px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,0.2);
        }
        [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
            padding-top: 0.75rem !important;
            padding-bottom: 0.75rem !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.4rem 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            gap: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p {
            margin-bottom: 4px !important;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 10px;
            padding: 14px 18px !important;
        }
        div[data-testid="stFormSubmitButton"] > button {
            background: #534AB7 !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #3D36A0 !important;
        }
        div[data-testid="stForm"] {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 10px;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _v(key: str) -> Any:
    return st.session_state.get(f"cr_{key}", SAMPLE_DEFAULTS.get(key, ""))


# ── Page: Create Run ──────────────────────────────────────────────────────────


def page_create_run() -> None:
    _section("Create Run")
    st.markdown(
        "Submit a new assessment generation job. "
        "The agent pipeline runs asynchronously — track progress in **All Runs**."
    )

    # Template loader row
    tcol, lcol = st.columns([3, 1])
    with tcol:
        tmpl_name = st.selectbox(
            "Industry template",
            list(INDUSTRY_TEMPLATES.keys()),
            help="Pre-fill the form from a validated industry template.",
        )
    with lcol:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load template", use_container_width=True):
            for k, v in INDUSTRY_TEMPLATES[tmpl_name].items():
                st.session_state[f"cr_{k}"] = v
            st.rerun()

    # Randomize row
    rcol, hcol = st.columns([1.6, 4])
    with rcol:
        if st.button("🎲 Randomize values", use_container_width=True):
            profile = random.choice(_RAND_PROFILES)
            for k in ("location", "language", "industry", "company_context"):
                st.session_state[f"cr_{k}"] = profile[k]
            for k, pool in RANDOMIZE_POOL.items():
                st.session_state[f"cr_{k}"] = random.choice(pool)
            st.session_state["cr_dimensions"] = ""
            st.session_state["cr_entities"] = ""
            st.rerun()
    with hcol:
        st.caption(
            "Fills all fields with a realistic randomized scenario drawn from the full "
            "options pool. Dimensions and entities are cleared — the agent auto-generates them."
        )

    st.divider()

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("create_run_form"):
        st.markdown("##### Core parameters")

        c1, c2, c3, c4 = st.columns(4)
        industry = c1.text_input("Industry *", value=_v("industry"))
        role = c2.text_input("Role *", value=_v("role"))
        location = c3.text_input("Location *", value=_v("location"))
        language = c4.text_input("Language *", value=_v("language"))

        c5, c6, c7, c8 = st.columns(4)
        cur_seniority = _v("seniority")
        seniority = c5.selectbox(
            "Seniority *",
            SENIORITY_OPTS,
            index=SENIORITY_OPTS.index(cur_seniority) if cur_seniority in SENIORITY_OPTS else 1,
        )
        cur_tool = _v("tool")
        tool = c6.selectbox(
            "Tool *",
            TOOL_OPTS,
            index=TOOL_OPTS.index(cur_tool) if cur_tool in TOOL_OPTS else 0,
        )
        duration_minutes = c7.number_input(
            "Duration (min) *",
            min_value=30,
            max_value=180,
            step=15,
            value=int(_v("duration_minutes") or 75),
        )
        cur_fmt = _v("deliverable_format")
        deliverable_format = c8.selectbox(
            "Format *",
            FORMAT_OPTS,
            index=FORMAT_OPTS.index(cur_fmt) if cur_fmt in FORMAT_OPTS else 0,
        )

        st.markdown("##### Company context *")
        company_context = st.text_area(
            "company_context",
            value=_v("company_context"),
            height=110,
            label_visibility="collapsed",
            placeholder="Describe the hiring company — industry, size, distribution channels, etc.",
        )

        st.markdown("##### Optional overrides")
        st.caption(
            "Leave both blank to let the agent auto-generate dimensions and entities "
            "from the industry context."
        )
        oc1, oc2 = st.columns(2)
        with oc1:
            dimensions_raw = st.text_area(
                "Dimensions (JSON)",
                value=_v("dimensions") or "",
                height=150,
                placeholder='{\n  "axis_name": ["option_a", "option_b"]\n}',
                help="Scoring axes for scenario rolling. JSON object. Blank = auto-inferred.",
            )
        with oc2:
            entities_raw = st.text_area(
                "Entities (one per line)",
                value=_v("entities") or "",
                height=150,
                placeholder="clients\ncontrats\ntransactions",
                help="Data table names to generate. One per line. Blank = auto-inferred.",
            )

        submitted = st.form_submit_button("Submit run →", type="primary", use_container_width=True)

    if submitted:
        missing = [
            f
            for f, v in [
                ("Industry", industry),
                ("Role", role),
                ("Location", location),
                ("Language", language),
                ("Tool", tool),
                ("Company context", company_context),
            ]
            if not v.strip()
        ]
        if missing:
            st.error(f"Missing required fields: {', '.join(missing)}")
            return

        dimensions = None
        if dimensions_raw.strip():
            try:
                dimensions = json.loads(dimensions_raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in Dimensions: {e}")
                return

        entities = None
        if entities_raw.strip():
            entities = [ln.strip() for ln in entities_raw.splitlines() if ln.strip()]

        payload: dict[str, Any] = {
            "industry": industry,
            "company_context": company_context,
            "location": location,
            "language": language,
            "role": role,
            "seniority": seniority,
            "tool": tool,
            "duration_minutes": int(duration_minutes),
            "deliverable_format": deliverable_format,
        }
        if dimensions:
            payload["dimensions"] = dimensions
        if entities:
            payload["entities"] = entities

        with st.spinner("Submitting…"):
            result, err = _api_post("/runs", payload)

        if err:
            st.error(err)
        else:
            run_id = result["run_id"]
            st.success(f"Run queued: **{run_id}**")
            st.info(
                f"Open **All Runs** to monitor progress, or jump to "
                f"**Run Detail** and enter `{run_id}`."
            )


# ── Page: All Runs ────────────────────────────────────────────────────────────


def page_all_runs() -> None:
    _section("All Runs")

    hcol, rcol, acol = st.columns([3, 1, 2])
    with rcol:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with acol:
        auto_refresh = st.checkbox("Auto-refresh every 10 s")

    data, err = _api_get("/runs")
    if err:
        st.error(err)
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        return

    runs = data.get("runs", [])
    if not runs:
        st.info("No runs yet. Create one in **Create Run**.")
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        return

    # Metrics
    statuses = [r["status"] for r in runs]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total", len(runs))
    mc2.metric("Done", statuses.count("done"))
    mc3.metric("In progress", statuses.count("running") + statuses.count("queued"))
    mc4.metric("Failed", statuses.count("failed"))
    st.divider()

    # Column headers
    hd = st.columns([1.2, 1.2, 1.8, 1.8, 1.8, 1.8, 0.8])
    labels = ["Run ID", "Status", "Role", "Industry", "Tool", "Location", ""]
    for col, lbl in zip(hd, labels, strict=True):
        col.markdown(
            f"<span style='font-size:12px;font-weight:700;opacity:0.55;"
            f"text-transform:uppercase'>{lbl}</span>",
            unsafe_allow_html=True,
        )
    st.divider()

    for r in reversed(runs):
        p = r.get("parameters") or {}
        rid = r["run_id"]
        cols = st.columns([1.2, 1.2, 1.8, 1.8, 1.8, 1.8, 0.8])
        cols[0].markdown(f"`{rid}`")
        cols[1].markdown(_badge(r["status"]), unsafe_allow_html=True)
        cols[2].markdown(f"{p.get('seniority', '—')} {p.get('role', '—')}")
        cols[3].markdown(p.get("industry", "—"))
        cols[4].markdown(p.get("tool", "—"))
        cols[5].markdown(p.get("location", "—"))
        if cols[6].button("→", key=f"goto_{rid}", help="Open detail view"):
            st.session_state["detail_run_id"] = rid
            st.session_state["_pending_nav"] = "Run Detail"
            st.rerun()

        scenario = r.get("scenario") or {}
        if scenario:
            with st.expander(f"Scenario — {rid}", expanded=False):
                scols = st.columns(min(len(scenario), 4))
                for i, (k, v) in enumerate(scenario.items()):
                    scols[i % 4].markdown(
                        f"<span style='font-size:11px;font-weight:700;color:#534AB7'>{k}</span>"
                        f"<br>{v}",
                        unsafe_allow_html=True,
                    )
        st.divider()

    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ── Page: Run Detail ──────────────────────────────────────────────────────────


def page_run_detail() -> None:
    _section("Run Detail")

    icol, _ = st.columns([2, 4])
    with icol:
        run_id = st.text_input(
            "Run ID",
            value=st.session_state.get("detail_run_id", ""),
            placeholder="run_001",
        )

    if not run_id.strip():
        st.info("Enter a run ID above to load its details.")
        return

    run_id = run_id.strip()
    st.session_state["detail_run_id"] = run_id

    data, err = _api_get(f"/runs/{run_id}")
    if err:
        st.error(err)
        return

    status = data["status"]
    params = data.get("parameters") or {}
    scenario = data.get("scenario") or {}
    mlflow_rid = data.get("mlflow_run_id")

    # Header
    hcol, scol, rfcol = st.columns([2, 1.5, 1])
    hcol.markdown(f"### {run_id}")
    scol.markdown(_badge(status), unsafe_allow_html=True)
    with rfcol:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    if mlflow_rid:
        st.caption(f"MLflow run: `{mlflow_rid}`")

    st.divider()

    # Parameters grid
    if params:
        st.markdown("**Parameters**")
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(_param("INDUSTRY", params.get("industry", "—")), unsafe_allow_html=True)
        p2.markdown(
            _param("ROLE", f"{params.get('seniority', '—')} {params.get('role', '—')}"),
            unsafe_allow_html=True,
        )
        p3.markdown(_param("TOOL", params.get("tool", "—")), unsafe_allow_html=True)
        p4.markdown(_param("LOCATION", params.get("location", "—")), unsafe_allow_html=True)
        st.markdown("")
        p5, p6, p7, _ = st.columns(4)
        p5.markdown(_param("LANGUAGE", params.get("language", "—")), unsafe_allow_html=True)
        p6.markdown(
            _param("DURATION", f"{params.get('duration_minutes', '—')} min"),
            unsafe_allow_html=True,
        )
        p7.markdown(
            _param("FORMAT", params.get("deliverable_format", "—")),
            unsafe_allow_html=True,
        )

    # Scenario
    if scenario:
        st.divider()
        st.markdown("**Scenario**")
        scols = st.columns(min(len(scenario), 3))
        for i, (k, v) in enumerate(scenario.items()):
            scols[i % 3].markdown(
                f"<span style='font-size:11px;font-weight:700;color:#534AB7;"
                f"text-transform:uppercase'>{k}</span><br>{v}",
                unsafe_allow_html=True,
            )

    if status == "done":
        base = _base()

        # ── Content review ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Content**")
        tab_brief, tab_solution = st.tabs(["📄 Candidate Brief", "🔑 Recruiter Solution"])

        with tab_brief:
            brief_text, brief_err = _fetch_text(base, f"/runs/{run_id}/brief")
            if brief_err:
                st.error(brief_err)
            elif brief_text:
                st.markdown(brief_text)

        with tab_solution:
            sol_text, sol_err = _fetch_text(base, f"/runs/{run_id}/solution")
            if sol_err:
                st.error(sol_err)
            elif sol_text:
                st.markdown(sol_text)

        # ── Package downloads ──────────────────────────────────────────────────
        st.divider()
        st.markdown("**Packages**")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(
                "<span style='font-weight:600'>Recruiter package</span><br>"
                "<span style='font-size:13px;color:#5F5E5A'>Brief + answer key solution</span>",
                unsafe_allow_html=True,
            )
            st.link_button(
                "Download recruiter.zip",
                f"{_public_base()}/runs/{run_id}/recruiter.zip",
                use_container_width=True,
                type="primary",
            )
        with dc2:
            st.markdown(
                "<span style='font-weight:600'>Candidate package</span><br>"
                "<span style='font-size:13px;opacity:0.65'>"
                "Brief + starter kit (no solution)</span>",
                unsafe_allow_html=True,
            )
            st.link_button(
                "Download candidate.zip",
                f"{_public_base()}/runs/{run_id}/candidate.zip",
                use_container_width=True,
                type="secondary",
            )

        # ── Score a submission ─────────────────────────────────────────────────
        st.divider()
        st.markdown("**Score a submission**")
        st.caption("Upload the completed project ZIP to score it against the recruiter answer key.")

        uploaded = st.file_uploader(
            "Solution ZIP",
            type=["zip"],
            key=f"score_upload_{run_id}",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Score", key=f"score_btn_{run_id}", type="primary"):
            with st.spinner("Scoring…"):
                try:
                    r = requests.post(
                        f"{_base()}/runs/{run_id}/score",
                        files={"solution": (uploaded.name, uploaded.getvalue(), "application/zip")},
                        timeout=120,
                    )
                    if r.status_code == 200:
                        st.session_state[f"score_result_{run_id}"] = r.text
                    else:
                        st.error(f"Scoring failed (HTTP {r.status_code}): {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API.")
                except Exception as e:
                    st.error(str(e))

        result_key = f"score_result_{run_id}"
        if result_key in st.session_state:
            st.markdown(st.session_state[result_key])

    elif status in ("queued", "running"):
        st.info("Run is in progress. Packages will be available once complete.")

    elif status == "failed":
        st.error("This run failed. Check the agent logs for details.")


# ── Page: Search ──────────────────────────────────────────────────────────────


def page_search() -> None:
    _section("Semantic Search")
    st.markdown(
        "Search across completed runs using natural language. "
        "Queries are expanded via **HyDE** (Hypothetical Document Embedding) "
        "for richer semantic matching."
    )

    with st.form("search_form"):
        qcol, lcol = st.columns([5, 1])
        query = qcol.text_input(
            "Query",
            placeholder="Senior Data Analyst, P&C insurance, Power BI, Paris",
            label_visibility="collapsed",
        )
        limit = lcol.number_input(
            "Max", min_value=1, max_value=100, value=10, label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Search →", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a query to search.")
            st.session_state.pop("search_results", None)
        else:
            with st.spinner("Searching…"):
                results, err = _api_get("/runs/search", params={"q": query, "limit": int(limit)})
            if err:
                st.error(err)
                st.session_state.pop("search_results", None)
            else:
                st.session_state["search_results"] = results
                st.session_state["search_query"] = query

    results = st.session_state.get("search_results")
    if not results:
        if submitted and not st.session_state.get("search_results"):
            st.info("No results. Try different terms or ensure runs have been indexed.")
        return

    st.markdown(f"**{len(results)} result(s)** for: *{st.session_state.get('search_query', '')}*")
    st.divider()

    hd = st.columns([0.8, 1.2, 1.8, 1.8, 1.8, 1.2, 0.8])
    labels = ["Score", "Run ID", "Role", "Industry", "Tool", "Status", ""]
    for col, lbl in zip(hd, labels, strict=True):
        col.markdown(
            f"<span style='font-size:12px;font-weight:700;opacity:0.55;"
            f"text-transform:uppercase'>{lbl}</span>",
            unsafe_allow_html=True,
        )
    st.divider()

    for r in results:
        score = r.get("score", 0.0)
        p = r.get("parameters") or {}
        rid = r["run_id"]
        score_pct = int(score * 100)
        score_color = "#1D9E75" if score_pct >= 80 else "#534AB7" if score_pct >= 60 else "#F59E0B"
        cols = st.columns([0.8, 1.2, 1.8, 1.8, 1.8, 1.2, 0.8])
        cols[0].markdown(
            f'<b style="color:{score_color};font-size:16px">{score_pct}%</b>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(f"`{rid}`")
        cols[2].markdown(f"{p.get('seniority', '—')} {p.get('role', '—')}")
        cols[3].markdown(p.get("industry", "—"))
        cols[4].markdown(p.get("tool", "—"))
        cols[5].markdown(_badge(r["status"]), unsafe_allow_html=True)
        if cols[6].button("→", key=f"s_goto_{rid}", help="Open detail view"):
            st.session_state["detail_run_id"] = rid
            st.session_state["_pending_nav"] = "Run Detail"
            st.rerun()
        st.divider()


# ── Page: Simulate ───────────────────────────────────────────────────────────


def page_simulate() -> None:
    _section("Simulate Candidate")
    st.markdown(
        "Run an AI-powered candidate simulation against a completed assessment. "
        "The agent reads the brief and starter project, then fills in a solution "
        "at the chosen proficiency level. A random **alea** factor (daily energy) "
        "is drawn automatically to vary focus and answer depth within the tier."
    )

    # ── Launch form ───────────────────────────────────────────────────────────
    st.markdown("#### Launch simulation")

    fc1, fc2, fc3 = st.columns([2, 2, 2])
    run_id_input = fc1.text_input(
        "Source run ID *",
        placeholder="run_001",
        help="The completed assessment run to simulate against.",
    )
    proficiency_raw = fc2.slider(
        "Proficiency",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="0 = very junior · 1 = perfect expert. Leave at 0 to sample randomly in [0.30, 0.95].",
    )
    alea_raw = fc3.slider(
        "Alea (daily energy)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Shape of the day. Low = distracted · High = motivated. 0 = draw randomly.",
    )

    submitted = st.button("Launch simulation →", type="primary", use_container_width=False)

    if submitted:
        if not run_id_input.strip():
            st.error("Source run ID is required.")
        else:
            payload: dict[str, Any] = {"run_id": run_id_input.strip()}
            if proficiency_raw > 0.0:
                payload["proficiency"] = proficiency_raw
            if alea_raw > 0.0:
                payload["alea"] = alea_raw

            with st.spinner("Queuing simulation…"):
                result, err = _api_post("/simulations", payload)

            if err:
                st.error(err)
            else:
                sim_id = result["simulation_id"]
                st.success(f"Simulation queued: **{sim_id}**")
                st.session_state["sim_detail_id"] = sim_id
                st.info("Track progress in the **Check status** section below.")

    st.divider()

    # ── Status / download ─────────────────────────────────────────────────────
    st.markdown("#### Check status")
    dcol, _ = st.columns([2, 4])
    with dcol:
        sim_id_check = st.text_input(
            "Simulation ID",
            value=st.session_state.get("sim_detail_id", ""),
            placeholder="sim_001",
            key="sim_id_check_input",
        )

    if sim_id_check.strip():
        sid = sim_id_check.strip()
        st.session_state["sim_detail_id"] = sid

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

        data, err = _api_get(f"/simulations/{sid}")
        if err:
            st.error(err)
        else:
            status = data["status"]
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.markdown(_badge(status), unsafe_allow_html=True)
            prof = data.get("proficiency")
            hc2.markdown(
                _param("PROFICIENCY", f"{prof:.0%}" if prof is not None else "—"),
                unsafe_allow_html=True,
            )
            alea = data.get("alea")
            hc3.markdown(
                _param("ALEA", f"{alea:.2f}" if alea is not None else "—"),
                unsafe_allow_html=True,
            )
            hc4.markdown(
                _param("SOURCE RUN", data.get("source_run_id", "—")),
                unsafe_allow_html=True,
            )

            if data.get("mlflow_run_id"):
                st.caption(f"MLflow run: `{data['mlflow_run_id']}`")

            if status == "done":
                base = _base()
                st.divider()

                tab_score, tab_dl = st.tabs(["📊 Scoring", "📦 Download"])

                with tab_score:
                    scoring_text, scoring_err = _fetch_text(base, f"/simulations/{sid}/scoring")
                    if scoring_err:
                        st.warning(f"Scoring not available: {scoring_err}")
                    elif scoring_text:
                        try:
                            st.markdown(
                                ScoringResult.model_validate_json(scoring_text).to_markdown()
                            )
                        except Exception as _exc:
                            st.warning(f"Could not render scoring result: {_exc}")

                with tab_dl:
                    st.markdown("Download the completed starter project:")
                    st.link_button(
                        "Download solution.zip",
                        f"{_public_base()}/simulations/{sid}/solution.zip",
                        use_container_width=False,
                        type="primary",
                    )

            elif status in ("queued", "running"):
                st.info("Simulation in progress. Refresh to check for updates.")
            elif status == "failed":
                st.error("Simulation failed. Check the agent logs for details.")
    else:
        st.info("Enter a simulation ID above to check its status.")


# ── Page: Docs ────────────────────────────────────────────────────────────────


def _preprocess_readme(content: str, base_dir: Path) -> str:
    def replace_image(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        if path.startswith("http"):
            return str(m.group(0))
        img_path = base_dir / path
        if not img_path.exists():
            return ""
        mime = "image/svg+xml" if img_path.suffix == ".svg" else "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        style = "max-width:100%;height:auto"
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="{style}"/>'

    def neutralize_local_md_link(m: re.Match[str]) -> str:
        text, path = m.group(1), m.group(2)
        if path.startswith("http") or not path.endswith(".md"):
            return str(m.group(0))
        return text  # drop unservable local link, keep display text

    content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, content)
    return re.sub(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", neutralize_local_md_link, content)


def page_docs() -> None:
    _section("Documentation")
    if not _README.exists():
        st.warning("README.md not found.")
        return
    content = _README.read_text(encoding="utf-8")
    st.markdown(_preprocess_readme(content, _README.parent), unsafe_allow_html=True)

    spec_files = [
        ("🇺🇸 English", _APP_ROOT / "docs" / "specification_en.md"),
        ("🇫🇷 Français", _APP_ROOT / "docs" / "specification_fr.md"),
    ]
    available = [(label, path) for label, path in spec_files if path.exists()]
    if available:
        st.divider()
        _section("Specifications")
        tabs = st.tabs([label for label, _ in available])
        for tab, (_, path) in zip(tabs, available, strict=True):
            with tab:
                st.markdown(
                    _preprocess_readme(path.read_text(encoding="utf-8"), path.parent),
                    unsafe_allow_html=True,
                )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="BUC Factory",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        mark_svg = _LOGO_MARK.read_text(encoding="utf-8") if _LOGO_MARK.exists() else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin:12px 0 8px 0">'
            f'<div style="width:64px;flex-shrink:0">{mark_svg}</div>'
            f"<div>"
            f'<div style="font-size:20px;font-weight:700;line-height:1.2">'
            f"Use Case Factory</div>"
            f'<div style="font-size:12px;opacity:0.55;line-height:1.4">'
            f"Recruitment assessments at scale</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        _pending = st.session_state.pop("_pending_nav", None)
        if _pending in ["Create Run", "All Runs", "Run Detail", "Search", "Simulate", "Docs"]:
            st.session_state["nav"] = _pending
        nav = st.radio(
            "Navigation",
            ["Create Run", "All Runs", "Run Detail", "Search", "Simulate", "Docs"],
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("API SERVER")
        api_base = st.text_input(
            "Server URL",
            value=st.session_state.get(
                "api_base",
                os.environ.get(
                    "API_PUBLIC_URL", os.environ.get("API_BASE_URL", "http://localhost:8010")
                ),
            ),
            label_visibility="collapsed",
            key="api_base_input",
        )
        st.session_state["api_base"] = api_base

        _health_url = os.environ.get("API_BASE_URL", api_base).strip().rstrip("/")
        try:
            r = requests.get(f"{_health_url}/docs", timeout=2)
            if r.status_code < 400:
                st.markdown(
                    '<span style="color:#1D9E75;font-size:13px">● API online</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="color:#F59E0B;font-size:13px">● API error</span>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<span style="color:#EF4444;font-size:13px">● API offline</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("GITHUB REPO")
        gh_url = st.text_input(
            "GitHub URL",
            value="https://github.com/AllanTamdem/buc_factory",
            label_visibility="collapsed",
            key="gh_url_input",
        )

        try:
            gh_r = requests.get(gh_url, timeout=3)
            if gh_r.status_code == 200:
                st.markdown(
                    '<span style="color:#1D9E75;font-size:13px">● Repo reachable</span>',
                    unsafe_allow_html=True,
                )
            elif gh_r.status_code == 404:
                st.markdown(
                    '<span style="color:#9CA3AF;font-size:13px">● Repo private</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="color:#F59E0B;font-size:13px">● HTTP {gh_r.status_code}</span>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<span style="color:#EF4444;font-size:13px">● Unreachable</span>',
                unsafe_allow_html=True,
            )

    # ── Route ─────────────────────────────────────────────────────────────────
    if nav == "Create Run":
        page_create_run()
    elif nav == "All Runs":
        page_all_runs()
    elif nav == "Run Detail":
        page_run_detail()
    elif nav == "Search":
        page_search()
    elif nav == "Simulate":
        page_simulate()
    elif nav == "Docs":
        page_docs()


def run() -> None:
    """Console script entry point: launches streamlit run on this file."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:],
        check=False,
    )


if __name__ == "__main__":
    main()
