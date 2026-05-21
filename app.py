# app.py — CKM Graph RAG v2 Intelligent Clinical Decision Support Agent
import streamlit as st
import os
import math
import base64
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings

from modules.viz import (
    risk_gauge, trajectory_chart,
    counterfactual_bar, risk_score_card, risk_timeline,
    cox_contribution_waterfall,
    clpm_path_diagram, shap_lollipop,
    trajectory_phenotype_chart, radar_phenotype,
    ode_trajectory_chart, external_auc_chart,
    bibliometrics_panel,
)
from modules.predictor import predict as py_predict, risk_level as py_risk_level
try:
    from modules.predictor_v6 import predict_v6, is_v6_available, build_static_from_v13_inputs
    _V6_IMPORTABLE = True
except ImportError:
    _V6_IMPORTABLE = False
    def is_v6_available(): return False
from modules import ocr as ocr_module
from modules.science import load_index, recommend_resources, render_resource_grid
from modules.admin import check_auth, render_video_upload, render_kb_upload, render_video_management

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CKM Graph RAG v2", page_icon="🧬", layout="wide")

# ─── Constants ────────────────────────────────────────────────────────────────
ZHIPU_KEY  = "1cddba76ebff472d97774d5b55fabd3c.s1DzJye0WJKfbaDg"
os.environ["ZHIPUAI_API_KEY"] = ZHIPU_KEY
client     = OpenAI(api_key=ZHIPU_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")
MODEL_NAME = "glm-4"

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Bibliometrics hardcoded demo stats ───────────────────────────────────────
BIBLIO_STATS = {
    "years":          list(range(2018, 2026)),
    "counts":         [12, 18, 24, 31, 45, 67, 89, 112],
    "keywords":       ["CKM syndrome", "cardiovascular risk", "kidney disease",
                       "metabolic syndrome", "SGLT2 inhibitor", "eGFR", "TyG index",
                       "trajectory", "GBTM", "RAASi", "heart failure", "diabetes",
                       "hypertension", "biomarker", "prediction"],
    "keyword_counts": [89, 76, 71, 68, 54, 52, 48, 43, 38, 35, 34, 31, 29, 27, 24],
    "journals":       ["JACC", "Lancet Digital Health", "NEJM", "Circulation",
                       "KI", "JASN", "Diabetes Care", "Others"],
    "journal_counts": [18, 15, 12, 11, 9, 8, 7, 32],
    "total_papers":   398,
    "date_range":     "2018–2025",
    "top_journal":    "JACC",
}

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }

  .stTabs [aria-selected="true"] {
    background: #2980B9 !important;
    color: #FFFFFF !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    border-radius: 8px;
  }

  .risk-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
  }
  .risk-low   { background:#D5F5E3; color:#1A5C35; }
  .risk-mid   { background:#FEF3C7; color:#7C4B00; }
  .risk-high  { background:#FDEBD0; color:#9A3200; }
  .risk-vhigh { background:#FADBD8; color:#8B0000; }

  .metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    margin-bottom: 12px;
    border: 1px solid #D5D8DC;
  }

  .section-title {
    font-size: 11px;
    font-weight: 700;
    color: #5D6D7E;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
  }

  .stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #FFFFFF !important;
    background-color: #2980B9 !important;
    border: none !important;
    transition: all 0.15s ease;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  hr { border-color: #D5D8DC; }

  .biblio-card {
    background: #F8FAFF;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #BBCFEE;
    margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session state defaults ───────────────────────────────────────────────────
for key, default in {
    "prediction_done":        False,
    "result":                 None,
    "patient_data":           None,
    "raw_inputs":             None,
    "chat_history":           [],
    "system_context":         "",
    "comp_score":             None,
    "risk_level":             None,
    "view_mode":              "doctor",
    "initial_report":         "",
    "initial_report_patient": "",
    "onboarding_complete":    False,
    "current_ip_state":       None,
    "show_mascot":            True,
    "show_story_modal":       False,
    "story_anchor":           None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Dynamic theme injection ───────────────────────────────────────────────────
def _inject_theme():
    if st.session_state.get("show_mascot", True):
        st.markdown("""
<style>
  .stApp { background-color: #FFF0F5 !important; }
  section[data-testid="stSidebar"] { background-color: #FCE4EC !important; }
  .stTabs [aria-selected="true"] { background: #D81B60 !important; color: #fff !important; }
  .stButton > button { background-color: #D81B60 !important; }
  .metric-card { background: #FFF8FB !important; border-color: #F8BBD9 !important; }
  hr { border-color: #F8BBD9 !important; }
</style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<style>
  .stApp { background-color: #F0F4FF !important; }
  section[data-testid="stSidebar"] { background-color: #E3EAF8 !important; }
  .stTabs [aria-selected="true"] { background: #1565C0 !important; color: #fff !important; }
  .stButton > button { background-color: #1565C0 !important; }
  .metric-card { background: #FFFFFF !important; border-color: #BBCFEE !important; }
  hr { border-color: #BBCFEE !important; }
</style>""", unsafe_allow_html=True)

_inject_theme()

# ─── Cached data loaders ──────────────────────────────────────────────────────
_DATA_DIR = os.path.join(_APP_DIR, "data")

@st.cache_data
def _load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, filename)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

traj_df = _load_csv("fig1_trajectory_profiles.csv")
shap_df = _load_csv("figure3_real_shap.csv")
clpm_df = _load_csv("clpm_key_paths_summary_v2.csv")
ode_df  = _load_csv("figure3_real_ode.csv")
ext_df  = _load_csv("fig5_external_forest.csv")

# ─── Cached vector DB ─────────────────────────────────────────────────────────
@st.cache_resource
def load_database():
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    return Chroma(persist_directory=os.path.join(_APP_DIR, "chroma_db"),
                  embedding_function=embeddings)

vector_db = load_database()

# ─── Clinical helpers ─────────────────────────────────────────────────────────
def calc_map(sbp, dbp):
    return (sbp + 2 * dbp) / 3.0

def calc_egfr_ckd_epi(scr_umol, age, gender):
    scr_mgdl  = scr_umol / 88.4
    is_female = gender == "Female"
    k         = 0.7 if is_female else 0.9
    a         = -0.329 if is_female else -0.411
    multiplier = 1.018 if is_female else 1.0
    min_val   = min(scr_mgdl / k, 1)
    max_val   = max(scr_mgdl / k, 1)
    egfr      = 141 * (min_val ** a) * (max_val ** -1.209) * (0.993 ** age) * multiplier
    return round(egfr, 2)

def calc_slope(series: list) -> float:
    """Linear regression slope over time points (per year)."""
    x = np.array(range(len(series)), dtype=float)
    y = np.array(series, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

def calc_comprehensive_score(mace_risk: float, death_risk: float,
                              ckm_approx: float, map_val: float,
                              bmi: float, htn: int, dm: int) -> tuple:
    """
    Composite risk score for v13 ensemble model.
    Combines model probabilities with CKM staging and cardiometabolic burden.
    """
    prob_norm  = mace_risk * 0.55 + death_risk * 0.45
    # CKM stage penalty (0-4 → 0-0.25)
    stage_pen  = min(ckm_approx / 4.0, 1.0) * 0.25
    # MAP burden (>100 mmHg adds penalty)
    map_pen    = max(0.0, min((map_val - 100) / 60.0, 1.0)) * 0.10
    # Comorbidity bonus
    comorbid   = (htn + dm) * 0.025
    score      = min(prob_norm + stage_pen * 0.3 + map_pen + comorbid, 1.0)

    if score < 0.05:    level = "Low"
    elif score < 0.10:  level = "Moderate"
    elif score < 0.20:  level = "High"
    else:               level = "Very High"

    return round(score, 4), level

def _rag_query(query: str, k: int = 4) -> str:
    """Layered RAG: guidelines → cases → rules."""
    docs = vector_db.similarity_search(query, k=k)
    return "\n".join(f"[Ref {i+1}]: {d.page_content}" for i, d in enumerate(docs))


# ─── IP Mascot System ─────────────────────────────────────────────────────────
_IP_ASSETS_DIR = os.path.join(_APP_DIR, "assets", "ip-states")
_IP_TOTAL_IMG  = os.path.join(_IP_ASSETS_DIR, "total.png")

IP_STATE_CONFIG = {
    "optimal":   {
        "img":   os.path.join(_IP_ASSETS_DIR, "optimal.png"),
        "alt":   "CKM Guardian - Optimal Function",
        "label": "Optimal Function",
        "desc":  "Your cardiometabolic risk profile is well balanced. Keep up the healthy habits!",
        "anchor": "ip-optimal",
    },
    "cardiac":   {
        "img":   os.path.join(_IP_ASSETS_DIR, "cardiac.png"),
        "alt":   "CKM Guardian - Cardiac Load",
        "label": "Cardiac Load",
        "desc":  "Your blood pressure is elevated, increasing cardiac workload. Blood pressure control is the priority.",
        "anchor": "ip-cardiac",
    },
    "metabolic": {
        "img":   os.path.join(_IP_ASSETS_DIR, "metabolic.png"),
        "alt":   "CKM Guardian - Metabolic Dysregulation",
        "label": "Metabolic Dysregulation",
        "desc":  "Elevated BMI or diabetes is driving metabolic stress — weight management and glucose control are key.",
        "anchor": "ip-metabolic",
    },
    "renal":     {
        "img":   os.path.join(_IP_ASSETS_DIR, "renal.png"),
        "alt":   "CKM Guardian - Renal Burden",
        "label": "Renal-Cardiac Burden",
        "desc":  "Combined hypertension and elevated CKM stage suggest kidney-heart crosstalk. Close monitoring needed.",
        "anchor": "ip-renal",
    },
    "crisis":    {
        "img":   os.path.join(_IP_ASSETS_DIR, "crisis.png"),
        "alt":   "CKM Guardian - Crisis",
        "label": "High-Risk Crisis",
        "desc":  "MACE or mortality risk is critically elevated. Urgent clinical review and intensive intervention required.",
        "anchor": "ip-crisis",
    },
    "sglt2":     {
        "img":   os.path.join(_IP_ASSETS_DIR, "sglt2.png"),
        "alt":   "CKM Guardian - SGLT2 Protection",
        "label": "SGLT2 Protection",
        "desc":  "An SGLT2 inhibitor is actively protecting your heart and kidneys — continue adherence.",
        "anchor": "ip-sglt2",
    },
}


def _load_img_b64(path: str):
    """Read an image file and return its base64 string; returns None if file not found."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def update_ip_state(
    mace_risk: float,
    death_risk: float,
    ckm_approx: float,
    map_val: float,
    bmi: float,
    sbp: float,
    htn: int,
    dm: int,
    report_text: str = "",
) -> str:
    """
    Determine IP state from v13 model outputs.
    Priority: sglt2 > crisis > renal > cardiac > metabolic > optimal

    v13 variables replace old trajectory series:
    - crisis:    mace_risk > 0.15 OR death_risk > 0.10 OR ckm_approx >= 3.5
    - renal:     ckm_approx >= 2.5 AND (htn or sbp >= 140)  [renal-cardiac overlap]
    - cardiac:   map_val >= 107 OR sbp >= 140 OR htn=1 with high MAP
    - metabolic: bmi >= 28 OR dm=1
    - optimal:   everything else
    """
    sglt2_keywords = ["sglt2", "empagliflozin", "dapagliflozin", "canagliflozin",
                      "listenet", "engliflozin", "dagliflozin", "cagliflozin"]
    if any(kw in report_text.lower() for kw in sglt2_keywords):
        return "sglt2"
    if mace_risk > 0.15 or death_risk > 0.10 or ckm_approx >= 3.5:
        return "crisis"
    if ckm_approx >= 2.5 and (htn or sbp >= 140):
        return "renal"
    if map_val >= 107 or sbp >= 140 or (htn and map_val >= 100):
        return "cardiac"
    if bmi >= 28 or dm:
        return "metabolic"
    return "optimal"


def _render_ip_cover():
    """Show full group photo or six individual images before analysis."""
    st.markdown("<div style='text-align:center;margin:32px 0 16px;'>", unsafe_allow_html=True)
    total_b64 = _load_img_b64(_IP_TOTAL_IMG)
    if total_b64:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<img src='data:image/png;base64,{total_b64}' "
            f"     alt='CKM Six Health Guardians' width='280' "
            f"     style='border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,0.10);'/>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center;font-size:11px;color:#95A5A6;margin-top:6px;'>"
            "All six guardians are here! 👋</div>",
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(6)
        for i, (state, cfg) in enumerate(IP_STATE_CONFIG.items()):
            b64 = _load_img_b64(cfg["img"])
            with cols[i]:
                if b64:
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<img src='data:image/png;base64,{b64}' "
                        f"     alt='{cfg['alt']}' width='70' "
                        f"     style='border-radius:8px;'/>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='text-align:center;font-size:20px;'>❓</div>",
                                unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;font-size:14px;color:#5D6D7E;margin-top:12px;font-weight:500;'>"
        "Your CKM Health Guardians — always by your side 💪</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("📖 Meet Your Guardians", key="open_story_cover"):
        st.session_state["show_story_modal"] = True
        st.session_state["story_anchor"]     = None
        st.rerun()


def _render_ip_dynamic():
    """Show current IP state at the top of the results area after analysis."""
    state = st.session_state.get("current_ip_state")
    if not state or state not in IP_STATE_CONFIG:
        return

    cfg = IP_STATE_CONFIG[state]
    b64 = _load_img_b64(cfg["img"])

    col_img, col_info = st.columns([1, 4])
    with col_img:
        if b64:
            st.markdown(
                f"<img src='data:image/png;base64,{b64}' "
                f"     alt='{cfg['alt']}' width='100' "
                f"     style='border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,0.10);'/>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("🩺", unsafe_allow_html=True)
    with col_info:
        st.markdown(
            f"<div style='font-size:13px;color:#2C3E50;line-height:1.6;margin-top:8px;'>"
            f"{cfg['desc']}</div>",
            unsafe_allow_html=True,
        )
        if st.button("❓ Learn More", key="open_story_dynamic"):
            st.session_state["show_story_modal"] = True
            st.session_state["story_anchor"]     = cfg["anchor"]
            st.rerun()


def _render_story_modal():
    """Companion mode: patient-story.md with IP images. Minimal mode: patient-science.md."""
    if not st.session_state.get("show_story_modal"):
        return

    is_companion = st.session_state.get("show_mascot", True)
    _md_path = os.path.join(_APP_DIR, "assets",
                            "patient-story.md" if is_companion else "patient-science.md")
    try:
        with open(_md_path, encoding="utf-8") as f:
            story_md = f.read()
    except FileNotFoundError:
        story_md = f"File not found: {_md_path}"

    if is_companion:
        for state, cfg in IP_STATE_CONFIG.items():
            b64 = _load_img_b64(cfg["img"])
            if not b64:
                continue
            story_md = story_md.replace(
                f"![{cfg['alt']}](assets/ip-states/{state}.png)",
                f'<img src="data:image/png;base64,{b64}" alt="{cfg["alt"]}" width="220" style="border-radius:14px;margin:10px 0;" />',
            )
            story_md = story_md.replace(
                f'src="assets/ip-states/{state}.png"',
                f'src="data:image/png;base64,{b64}"',
            )
        total_b64 = _load_img_b64(_IP_TOTAL_IMG)
        if total_b64:
            story_md = story_md.replace(
                'src="assets/ip-states/total.png"',
                f'src="data:image/png;base64,{total_b64}"',
            )
    else:
        _network_path = os.path.join(_IP_ASSETS_DIR, "Network.png")
        _network_b64  = _load_img_b64(_network_path)
        if _network_b64:
            story_md = story_md.replace(
                'src="assets/ip-states/Network.png"',
                f'src="data:image/png;base64,{_network_b64}"',
            )

    anchor = st.session_state.get("story_anchor")

    with st.container():
        st.markdown("---")
        col_title, col_close = st.columns([5, 1])
        with col_title:
            if is_companion:
                st.markdown("### 📖 Meet Your Health Guardians")
            else:
                st.markdown("### 📄 CKM Syndrome: Pathobiologic Mechanisms & Evidence Review")
        with col_close:
            if st.button("✕ Close", key="close_story"):
                st.session_state["show_story_modal"] = False
                st.rerun()

        if anchor and is_companion:
            _lbl = IP_STATE_CONFIG.get(anchor.replace("ip-", ""), {}).get("label", "")
            st.info(f"💡 Scrolled to: {_lbl}")

        st.markdown(story_md, unsafe_allow_html=True)
        st.markdown("---")


# ─── Onboarding wizard ────────────────────────────────────────────────────────
_ONBOARDING_STEPS = [
    {
        "title": "🎨 Welcome — Choose Your Interface Style",
        "body":  "__STYLE_SELECT__",
        "icon":  "🎨",
    },
    {
        "title": "👋 Welcome to CKM Graph RAG v2",
        "body": "Start by uploading your patient's annual health check reports in the left panel. "
                "Images and PDFs are both supported. You can upload up to 5 years of reports at once.",
        "icon": "🏥",
    },
    {
        "title": "👁️ Dual View Modes",
        "body": "Switch between **Doctor View** and **Patient View** at the top of the page. "
                "Doctor View shows full quantitative data and clinical metrics. "
                "Patient View translates findings into plain language for direct patient communication.",
        "icon": "🔀",
    },
    {
        "title": "📊 Dynamic Risk Assessment",
        "body": "The system uses a v13 LightGBM + XGBoost + Logistic Regression ensemble model "
                "(19 engineered features) to compute 5-year MACE and all-cause mortality risk. "
                "Results include a composite risk score, CKM approximate stage, and trend charts.",
        "icon": "📈",
    },
    {
        "title": "🔬 Evidence-Based Intervention",
        "body": "In the **Counterfactual Simulation** tab you can adjust a single variable "
                "(e.g. SBP, BMI, DM status) and instantly see how the predicted risk changes — "
                "powered by the same v13 ensemble model with modified inputs.",
        "icon": "⚗️",
    },
    {
        "title": "💬 Clinical Dialogue & Education",
        "body": "Use the **Clinical Dialogue** tab to ask follow-up questions grounded in "
                "the patient's data and the latest guidelines. "
                "The **Health Education** tab auto-recommends videos tailored to this patient's risk profile.",
        "icon": "🎓",
    },
]


def _render_onboarding():
    """Render onboarding wizard; step 0 is interface style selection."""
    if "ob_step" not in st.session_state:
        st.session_state["ob_step"] = 0

    _step  = st.session_state["ob_step"]
    _total = len(_ONBOARDING_STEPS)
    _info  = _ONBOARDING_STEPS[_step]

    _lpad, _center, _rpad = st.columns([1, 2, 1])
    with _center:
        st.markdown(f"## {_info['icon']}  {_info['title']}")

        if _info["body"] == "__STYLE_SELECT__":
            st.markdown("Choose your preferred interface style. You can change this anytime in the sidebar.")
            _current = "Companion" if st.session_state.get("show_mascot", True) else "Professional"
            _style = st.radio(
                "Interface Style",
                ["Companion", "Professional"],
                index=0 if _current == "Companion" else 1,
                key="ob_style_radio",
                horizontal=True,
            )
            if _style == "Companion":
                st.info("🌸 **Companion Style**: Shows CKM Health Guardian mascots alongside clinical data to engage patients.")
            else:
                st.info("📊 **Professional Style**: Displays core data and charts only — ideal for clinical settings.")
            st.session_state["show_mascot"] = (_style == "Companion")
        else:
            st.info(_info['body'])

        st.progress((_step + 1) / _total)
        st.caption(f"Step {_step + 1} of {_total}")
        st.markdown("---")

    _bl, _b1, _b2, _br = st.columns([1, 1, 1, 1])
    with _b1:
        if st.button("⏭ Skip Tutorial", key="ob_skip", use_container_width=True):
            st.session_state["onboarding_complete"] = True
            st.session_state.pop("ob_step", None)
            st.rerun()
    with _b2:
        _lbl = "🚀 Get Started!" if _step == _total - 1 else "Next →"
        if st.button(_lbl, key="ob_next", type="primary", use_container_width=True):
            if _step < _total - 1:
                st.session_state["ob_step"] = _step + 1
                st.rerun()
            else:
                st.session_state["onboarding_complete"] = True
                st.session_state.pop("ob_step", None)
                st.rerun()


# ─── Guard: only render main UI when onboarding is done ──────────────────────
if not st.session_state.get("onboarding_complete", False):
    _render_onboarding()
    st.stop()

# ─── Page header + view toggle ────────────────────────────────────────────────
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0F4C81 0%,#1a6fba 60%,#2196F3 100%);
                padding:28px 32px;border-radius:16px;margin-bottom:20px;
                box-shadow:0 4px 24px rgba(15,76,129,0.25);'>
      <div style='display:flex;align-items:center;gap:12px;'>
        <div style='background:rgba(255,255,255,0.15);border-radius:12px;
                    padding:10px 14px;font-size:26px;line-height:1;'>🧬</div>
        <div>
          <div style='color:white;font-size:22px;font-weight:700;letter-spacing:-0.3px;'>
            CKM Graph RAG v2 — Intelligent Clinical Decision Support
          </div>
          <div style='color:rgba(255,255,255,0.65);font-size:12px;margin-top:4px;
                      font-weight:500;letter-spacing:0.3px;'>
            v13 Ensemble (LGB+XGB+LR) &nbsp;·&nbsp; Graph RAG &nbsp;·&nbsp;
            CLPM Causal Analysis &nbsp;·&nbsp; SHAP &nbsp;·&nbsp;
            ODE Simulation &nbsp;·&nbsp; Bibliometrics
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    mode = st.radio(
        "View mode",
        ["Doctor", "Patient"],
        horizontal=True,
        index=0 if st.session_state["view_mode"] == "doctor" else 1,
        label_visibility="collapsed",
    )
    st.session_state["view_mode"] = mode.lower()
    view_mode = st.session_state["view_mode"]
    is_doctor   = view_mode == "doctor"
    badge_bg    = "#EFF6FF" if is_doctor else "#F0FDF4"
    badge_color = "#1D4ED8" if is_doctor else "#166534"
    badge_text  = "👨‍⚕️ Doctor View" if is_doctor else "🧑 Patient View"
    st.markdown(
        f"<div style='text-align:center;font-size:12px;color:{badge_color};"
        f"font-weight:700;background:{badge_bg};padding:6px 10px;"
        f"border-radius:8px;margin-top:6px;'>{badge_text}</div>",
        unsafe_allow_html=True,
    )

view_mode = st.session_state["view_mode"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Patient Input & Prediction",
    "📊 Visual Analytics Dashboard",
    "🔬 Counterfactual Simulation",
    "💬 Clinical Dialogue",
    "📚 Health Education & Video",
    "📖 Literature & Bibliometrics",
    "🔧 Admin",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Input & Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("📋 Patient Data Entry")

        with st.expander("📎 Auto-fill from Medical Reports (OCR)", expanded=False):
            st.caption("Upload up to 5 annual health reports (images or PDFs). Assign each file to the correct year — gaps are supported.")
            uploaded_files = st.file_uploader(
                "Upload exam reports (images or PDFs, up to 5 files)",
                type=["jpg", "jpeg", "png", "pdf"],
                accept_multiple_files=True,
                key="ocr_upload",
            )
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.warning("At most 5 files allowed. Only the first 5 will be used.")
                    uploaded_files = uploaded_files[:5]

                st.markdown("**Assign each report to a year:**")
                year_assignments = {}
                used_years = []
                year_options = ["Y1", "Y2", "Y3", "Y4", "Y5"]
                assign_cols = st.columns(len(uploaded_files))
                for idx, f in enumerate(uploaded_files):
                    with assign_cols[idx]:
                        default_idx = 5 - len(uploaded_files) + idx
                        sel = st.selectbox(
                            f"📄 {f.name[:18]}",
                            year_options,
                            index=default_idx,
                            key=f"ocr_year_{idx}",
                        )
                        year_assignments[idx] = sel
                        used_years.append(sel)

                if len(used_years) != len(set(used_years)):
                    st.warning("⚠️ Duplicate year assignments detected. Please assign each file to a unique year.")

            if uploaded_files and st.button("🔍 Extract & Auto-fill", key="ocr_btn"):
                with st.spinner(f"Calling ZhipuAI Vision model for {len(uploaded_files)} report(s)..."):
                    try:
                        results = ocr_module.extract_from_files(uploaded_files, ZHIPU_KEY)
                        year_data = {}
                        for idx, r in enumerate(results):
                            yr = year_assignments.get(idx, year_options[idx])
                            year_data[yr] = r
                        st.session_state["ocr_year_data"] = year_data
                        errors = [r.get("_error") for r in results if r.get("_error")]
                        if errors:
                            st.warning(f"Some files failed: {'; '.join(errors)}")
                        else:
                            filled = ", ".join(sorted(year_data.keys()))
                            st.success(f"Extracted successfully! Filled: {filled}")
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

        ocr_year_data = st.session_state.get("ocr_year_data", {})
        _latest_yr = max(ocr_year_data.keys(), default=None)
        ocr = ocr_year_data.get(_latest_yr, {}) if _latest_yr else {}

        st.markdown("**Demographics:**")
        c_age, c_sex = st.columns(2)
        with c_age:
            age_input = st.number_input("Age (years):", min_value=18, max_value=100,
                                        value=int(ocr.get("age") or 55))
        with c_sex:
            gender_options = ["Male", "Female"]
            ocr_gender = ocr.get("gender")
            gender_default = gender_options.index(ocr_gender) if ocr_gender in gender_options else 0
            gender_input = st.selectbox("Gender:", gender_options, index=gender_default)

        st.markdown("---")
        st.markdown("**Cardiometabolic Measurements** *(current / most recent values)*:")

        c_bmi, c_sbp = st.columns(2)
        with c_bmi:
            bmi_input = st.number_input("⚖️ BMI (kg/m²):",
                                        min_value=15.0, max_value=60.0,
                                        value=float(ocr.get("bmi") or 25.0), step=0.1)
        with c_sbp:
            sbp_input = st.number_input("💓 SBP (mmHg):",
                                        min_value=70, max_value=250,
                                        value=int(ocr.get("sbp") or 125))

        c_dbp, c_map = st.columns(2)
        with c_dbp:
            dbp_input = st.number_input("💓 DBP (mmHg):",
                                        min_value=40, max_value=150,
                                        value=int(ocr.get("dbp") or 80))
        with c_map:
            _map_display = (sbp_input + 2 * dbp_input) / 3.0
            st.metric("MAP (auto)", f"{_map_display:.1f} mmHg")

        st.markdown("**Comorbidities:**")
        c_htn, c_dm, c_smk = st.columns(3)
        with c_htn:
            htn_input = st.checkbox("Hypertension", value=bool(sbp_input >= 140))
        with c_dm:
            dm_input = st.checkbox("Diabetes (DM)", value=False)
        with c_smk:
            smoking_input = st.checkbox("Current Smoker", value=False)

        st.markdown("---")
        st.markdown("**Optional: Historical Trend** *(3 time-points, oldest → newest)*:")
        st.caption("Used for trend charts only — does not affect the prediction model.")

        with st.expander("📈 Enter historical SBP / DBP / BMI trend", expanded=False):
            def _build_series(field: str, current_val: float) -> str:
                year_map = {"Y1": 0, "Y2": 1, "Y3": 2}
                series = [current_val - 10, current_val - 5, current_val]
                for yr, r in ocr_year_data.items():
                    idx = year_map.get(yr)
                    if idx is not None and r.get(field) is not None:
                        series[idx] = float(r[field])
                return ", ".join(f"{v:.1f}" for v in series)

            sbp_trend_input = st.text_input("SBP trend (3 values):",
                                            _build_series("sbp", float(sbp_input)))
            dbp_trend_input = st.text_input("DBP trend (3 values):",
                                            _build_series("dbp", float(dbp_input)))
            bmi_trend_input = st.text_input("BMI trend (3 values):",
                                            _build_series("bmi", bmi_input))

        st.markdown("---")
        follow_time = st.slider("⏱️ Prediction horizon (months):",
                                min_value=12, max_value=60, value=36, step=12)

        # Model selector
        _v6_ready = _V6_IMPORTABLE and is_v6_available()
        if _v6_ready:
            model_choice = st.radio(
                "🧠 Prediction model:",
                ["v13 Ensemble (fast, 8 inputs)", "V6 Deep Ensemble (longitudinal, AUC 0.88)"],
                index=0, horizontal=True,
                help="v13: LGB+XGB+LR, 19 features. V6: Transformer+BiLSTM+KAN+XGB+LGB, 626 features."
            )
        else:
            model_choice = "v13 Ensemble (fast, 8 inputs)"
            if _V6_IMPORTABLE:
                st.caption("V6 model: checkpoint not found — run boost_auc_v6.py to train.")

        analyze_btn = st.button("🚀 Start Intelligent Analysis", type="primary", use_container_width=True)

    with col_out:
        st.subheader("📊 Analysis & Clinical Report")

        if analyze_btn:
            _use_v6 = "V6" in model_choice
            _spinner_msg = "Running V6 deep ensemble..." if _use_v6 else "Running v13 ensemble prediction..."
            with st.spinner(_spinner_msg):
                try:
                    sex_male = 1 if gender_input == "Male" else 0
                    htn_val  = int(htn_input or sbp_input >= 140)
                    dm_val   = int(dm_input)
                    smk_val  = int(smoking_input)

                    def _parse_trend(s, fallback):
                        try:
                            vals = [float(x.strip()) for x in s.split(",")]
                            return vals if len(vals) >= 2 else fallback
                        except Exception:
                            return fallback

                    sbp_list = _parse_trend(sbp_trend_input,
                                            [sbp_input - 10, sbp_input - 5, float(sbp_input)])
                    dbp_list = _parse_trend(dbp_trend_input,
                                            [dbp_input - 5, dbp_input - 2, float(dbp_input)])
                    bmi_list = _parse_trend(bmi_trend_input,
                                            [bmi_input - 1.0, bmi_input - 0.5, bmi_input])
                    map_list = [calc_map(s, d) for s, d in zip(sbp_list, dbp_list)]

                    if _use_v6:
                        # Build yearly sequence from trend data
                        yearly = []
                        for sbp_y, dbp_y, bmi_y in zip(sbp_list, dbp_list, bmi_list):
                            map_y = (sbp_y + 2 * dbp_y) / 3.0
                            yearly.append({
                                'sbp': sbp_y, 'dbp': dbp_y, 'map': map_y,
                                'bmi': bmi_y, 'htn': htn_val, 'dm': dm_val,
                                'fpg': float(ocr.get('fpg') or 5.5),
                                'tg':  float(ocr.get('tg')  or 1.5),
                                'hdl': float(ocr.get('hdl') or 1.2),
                                'ldl': float(ocr.get('ldl') or 2.8),
                                'egfr': float(ocr.get('egfr') or 80.0),
                                'tyg': float(ocr.get('tyg') or 8.5),
                                'tx_bp_ever': htn_val,
                                'tx_dm_ever': dm_val,
                                'tx_lipid_ever': 0,
                            })
                        static_feats = build_static_from_v13_inputs(
                            age_input, sex_male, bmi_input,
                            float(sbp_input), float(dbp_input),
                            htn_val, dm_val, smk_val)
                        v6_res = predict_v6(yearly, static_feats)
                        if not v6_res.get('ok'):
                            st.error(f"V6 model error: {v6_res.get('error')}")
                            st.stop()
                        # Map V6 output to common result format
                        result = {
                            'ok':         True,
                            'mace_risk':  v6_res['mace_risk'],
                            'death_risk': v6_res['mace_risk'] * 0.6,  # proxy: no separate death model in V6
                            'map':        round((float(sbp_input) + 2 * float(dbp_input)) / 3.0, 1),
                            'ckm_approx': static_feats.get('ckm_stage', 0.0),
                            'model':      'v6_ensemble',
                            'mace_nn':    v6_res.get('mace_nn'),
                            'mace_meta':  v6_res.get('mace_meta'),
                        }
                    else:
                        result = py_predict(
                            age=age_input,
                            sex_male=sex_male,
                            bmi=bmi_input,
                            sbp=float(sbp_input),
                            dbp=float(dbp_input),
                            htn=htn_val,
                            dm=dm_val,
                            smoking=smk_val,
                        )

                    if not result.get("ok"):
                        st.error(f"Model error: {result.get('error', 'Unknown')}")
                        st.stop()

                    mace_risk  = result["mace_risk"]
                    comp_risk  = result["death_risk"]
                    map_val    = result["map"]
                    ckm_approx = result["ckm_approx"]
                    group_label = f"CKM Stage ~{ckm_approx:.0f}"

                    sbp_slope = calc_slope(sbp_list)
                    map_slope = calc_slope(map_list)
                    bmi_slope = calc_slope(bmi_list)

                    comp_score, risk_level = calc_comprehensive_score(
                        mace_risk, comp_risk, ckm_approx, map_val,
                        bmi_input, htn_val, dm_val)

                    patient_data = {
                        "sbp": sbp_list, "dbp": dbp_list, "bmi": bmi_list,
                        "map": map_list, "follow_time": follow_time,
                    }

                    st.session_state.update({
                        "prediction_done": True,
                        "result":          result,
                        "patient_data":    patient_data,
                        "age_input":       age_input,
                        "gender_input":    gender_input,
                        "comp_score":      comp_score,
                        "risk_level":      risk_level,
                        "sbp_slope":       sbp_slope,
                        "map_slope":       map_slope,
                        "bmi_slope":       bmi_slope,
                        "raw_inputs": {
                            "sbp": sbp_list, "dbp": dbp_list, "bmi": bmi_list,
                            "map": map_list, "follow_time": follow_time,
                            "htn": htn_val, "dm": dm_val, "smoking": smk_val,
                        },
                    })

                    _ip = update_ip_state(
                        mace_risk=mace_risk,
                        death_risk=comp_risk,
                        ckm_approx=ckm_approx,
                        map_val=map_val,
                        bmi=bmi_input,
                        sbp=float(sbp_input),
                        htn=htn_val,
                        dm=dm_val,
                    )
                    st.session_state["current_ip_state"] = _ip

                    if st.session_state.get("show_mascot", True):
                        _render_ip_dynamic()

                    st.plotly_chart(risk_gauge(mace_risk, comp_risk), use_container_width=True, key="gauge_new")

                    # V6 model detail badge
                    if result.get('model') == 'v6_ensemble':
                        nn_p  = result.get('mace_nn',   0.0)
                        met_p = result.get('mace_meta', 0.0)
                        st.info(
                            f"**V6 Deep Ensemble** (AUC ref 0.88) — "
                            f"NN: {nn_p:.3f} | Meta-LGB: {met_p:.3f} | "
                            f"Blended: {mace_risk:.3f}"
                        )

                    if view_mode == "patient":
                        _sbp_msg = (
                            "Your blood pressure is elevated — this increases your heart risk."
                            if sbp_input >= 140 else
                            "Your blood pressure is within an acceptable range."
                        )
                        _bmi_msg = (
                            "Your weight is above the healthy range — losing even 5% can reduce risk."
                            if bmi_input >= 28 else
                            "Your weight is in a healthy range."
                        )
                        _risk_color = {"Low": "#27AE60", "Moderate": "#F1C40F",
                                       "High": "#E67E22", "Very High": "#E74C3C"}.get(risk_level, "#2980B9")
                        st.markdown(f"""
                        <div style='background:white;border-radius:12px;padding:20px;
                                    box-shadow:0 2px 8px rgba(0,0,0,0.08);margin:12px 0;'>
                          <div style='font-size:13px;color:#7F8C8D;font-weight:600;
                                      text-transform:uppercase;letter-spacing:0.6px;'>
                            Your Health Summary
                          </div>
                          <div style='font-size:28px;font-weight:700;color:{_risk_color};margin:8px 0 4px;'>
                            {risk_level} Risk
                          </div>
                          <div style='font-size:14px;color:#2C3E50;line-height:1.7;'>
                            • {_sbp_msg}<br/>
                            • {_bmi_msg}<br/>
                            • Your overall risk score is <b>{comp_score*100:.1f}%</b>.
                              Your doctor will discuss a personalised plan with you.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.plotly_chart(
                        risk_score_card(comp_score, risk_level, mace_risk, comp_risk,
                                        sbp_slope, map_slope),
                        use_container_width=True, key="score_card_new",
                    )
                    st.plotly_chart(
                        risk_timeline(mace_risk, comp_risk, comp_score, risk_level),
                        use_container_width=True, key="timeline_new",
                    )
                    st.plotly_chart(trajectory_chart(patient_data), use_container_width=True, key="traj_new")

                    with st.spinner("Retrieving guidelines & generating structured report..."):
                        query = (f"CKM syndrome stage {ckm_approx:.0f} MACE risk {mace_risk:.2f} "
                                 f"mortality risk {comp_risk:.2f} MAP {map_val:.0f} mmHg "
                                 f"BMI {bmi_input:.1f} HTN={htn_val} DM={dm_val} intervention treatment")
                        ctx = _rag_query(query, k=4)

                        system_ctx = (
                            f"Patient: Age={age_input}, Gender={gender_input}, "
                            f"Follow-up={follow_time} months.\n"
                            f"v13 Ensemble: MACE Risk={mace_risk:.3f}, Mortality Risk={comp_risk:.3f}.\n"
                            f"CKM Stage≈{ckm_approx:.1f}, MAP={map_val:.1f} mmHg, "
                            f"BMI={bmi_input:.1f}, HTN={htn_val}, DM={dm_val}, Smoking={smk_val}.\n"
                            f"Composite Score={comp_score:.3f} ({risk_level} Risk).\n"
                            f"Guidelines context:\n{ctx}"
                        )
                        st.session_state["system_context"] = system_ctx
                        st.session_state["chat_history"]   = []

                        prompt_doctor = f"""You are a senior cardiologist. Generate a structured clinical intervention report in English.

## Patient Summary
- Age: {age_input}, Gender: {gender_input}, Follow-up: {follow_time} months
- **5-year MACE Risk: {mace_risk*100:.1f}%** | **5-year Mortality Risk: {comp_risk*100:.1f}%**
- Composite Risk Score: {comp_score*100:.1f}% ({risk_level} Risk)
- CKM Approximate Stage: {ckm_approx:.1f} | MAP: {map_val:.1f} mmHg
- BMI: {bmi_input:.1f} kg/m² | SBP: {sbp_input} mmHg | DBP: {dbp_input} mmHg
- Hypertension: {"Yes" if htn_val else "No"} | Diabetes: {"Yes" if dm_val else "No"} | Smoking: {"Yes" if smk_val else "No"}
- Model: v13 LightGBM + XGBoost + LogisticRegression ensemble (19 features)

## Evidence Base
{ctx}

## Required Output Structure (3-Step Prescription Engine)

### Step 1 — Risk Factor Targeting
For each elevated risk factor (BP, BMI, DM, smoking), state: current value, clinical target, and guideline rationale.

### Step 2 — Risk-Intensity Treatment Plan
Based on {risk_level} Risk and CKM Stage {ckm_approx:.0f}:
- Pharmacological interventions (drug class, dose range, monitoring)
- Lifestyle modifications (diet, exercise, weight targets)
- Intensity justification

### Step 3 — Monitoring & Follow-up Plan
- Frequency of assessments
- Trigger thresholds for escalation
- Patient education priorities

### Disclaimer
Add standard medical disclaimer.

Be specific, evidence-based, and concise."""

                        prompt_patient = f"""You are a caring doctor explaining health results to a patient in plain, friendly English.
The patient has NO medical background. Avoid all jargon.

## Patient's Results
- Overall risk level: {risk_level}
- Heart event risk over {follow_time} months: {mace_risk*100:.1f}%
- Mortality risk: {comp_risk*100:.1f}%
- Blood pressure: {sbp_input}/{dbp_input} mmHg {"(elevated)" if sbp_input >= 140 else "(acceptable)"}
- BMI: {bmi_input:.1f} {"(overweight)" if bmi_input >= 25 else "(healthy range)"}
- Diabetes: {"Yes" if dm_val else "No"} | Smoking: {"Yes" if smk_val else "No"}

## What the Guidelines Say (simplified)
{ctx}

## Write a patient-friendly health summary with these sections:

### What Your Results Mean
Explain the risk level in everyday language.

### What You Can Do
List 3–5 simple, actionable lifestyle changes. No drug names.

### What to Watch For
Simple warning signs to call their doctor.

### Your Next Steps
A short, encouraging closing paragraph.

Keep the whole response under 400 words. Be warm and reassuring."""

                        def _generate(prompt):
                            return client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "You are a professional medical AI assistant specializing in CKM syndrome."},
                                    {"role": "user",   "content": prompt},
                                ],
                                temperature=0.3,
                            ).choices[0].message.content

                        report_doctor  = _generate(prompt_doctor)
                        report_patient = _generate(prompt_patient)
                        st.session_state["initial_report"]         = report_doctor
                        st.session_state["initial_report_patient"] = report_patient

                    if view_mode == "doctor":
                        st.subheader("🤖 AI Evidence-Based Clinical Report")
                        st.markdown(report_doctor)
                    else:
                        st.subheader("📋 Your Personal Health Summary")
                        st.markdown(report_patient)

                except Exception as e:
                    st.error(f"Error: {e}")

        elif st.session_state["prediction_done"]:
            result      = st.session_state["result"]
            mace_risk   = result["mace_risk"]
            comp_risk   = result["death_risk"]
            comp_score  = st.session_state["comp_score"]
            risk_level  = st.session_state["risk_level"]
            sbp_slope   = st.session_state.get("sbp_slope", 0.0)
            map_slope   = st.session_state.get("map_slope", 0.0)

            if st.session_state.get("show_mascot", True):
                _render_ip_dynamic()

            st.plotly_chart(risk_gauge(mace_risk, comp_risk), use_container_width=True, key="gauge_cached")
            if view_mode == "patient":
                _raw = st.session_state.get("raw_inputs", {})
                _sbp_last = _raw.get("sbp", [125])[-1]
                _bmi_last = _raw.get("bmi", [25])[-1]
                _sbp_msg_c = ("Your blood pressure is elevated."
                              if _sbp_last >= 140 else
                              "Your blood pressure is within an acceptable range.")
                _bmi_msg_c = ("Your weight is above the healthy range."
                              if _bmi_last >= 28 else
                              "Your weight is in a healthy range.")
                _rc = {"Low": "#27AE60", "Moderate": "#F1C40F",
                       "High": "#E67E22", "Very High": "#E74C3C"}.get(risk_level, "#2980B9")
                st.markdown(f"""
                <div style='background:white;border-radius:12px;padding:20px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);margin:12px 0;'>
                  <div style='font-size:28px;font-weight:700;color:{_rc};margin-bottom:8px;'>
                    {risk_level} Risk
                  </div>
                  <div style='font-size:14px;color:#2C3E50;line-height:1.7;'>
                    • {_sbp_msg_c}<br/>• {_bmi_msg_c}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.plotly_chart(
                risk_score_card(comp_score, risk_level, mace_risk, comp_risk,
                                sbp_slope, map_slope),
                use_container_width=True, key="score_card_cached",
            )
            st.plotly_chart(
                risk_timeline(mace_risk, comp_risk, comp_score, risk_level),
                use_container_width=True, key="timeline_cached",
            )
            st.plotly_chart(trajectory_chart(st.session_state["patient_data"]), use_container_width=True, key="traj_cached")
            if view_mode == "doctor":
                st.subheader("🤖 AI Evidence-Based Clinical Report")
                st.markdown(st.session_state.get("initial_report", ""))
            else:
                st.subheader("📋 Your Personal Health Summary")
                st.markdown(st.session_state.get("initial_report_patient", ""))
        else:
            if st.session_state.get("show_mascot", True):
                _render_ip_cover()
            else:
                st.info("Fill in patient data on the left and click **Start Intelligent Analysis**.")

    _render_story_modal()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visual Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Visual Analytics Dashboard")
    section = st.radio(
        "Section",
        ["Trajectory Phenotypes", "CLPM Causal Paths", "SHAP Feature Importance", "ODE Intervention Simulation"],
        horizontal=True,
        key="viz_section",
    )

    if section == "Trajectory Phenotypes":
        if traj_df.empty:
            st.warning("Data file `fig1_trajectory_profiles.csv` not found in `data/` directory.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(trajectory_phenotype_chart(traj_df), use_container_width=True)
            with col2:
                st.plotly_chart(radar_phenotype(traj_df), use_container_width=True)

    elif section == "CLPM Causal Paths":
        if clpm_df.empty:
            st.warning("Data file `clpm_key_paths_summary_v2.csv` not found in `data/` directory.")
        else:
            st.plotly_chart(clpm_path_diagram(clpm_df), use_container_width=True)

            # Heatmap of cross-lagged betas by trajectory group
            st.markdown("#### Cross-Lagged Beta Coefficients by Trajectory Group")
            try:
                import plotly.graph_objects as _go_hm
                cl_paths = clpm_df[clpm_df["path_type"] != "AR"].copy()
                if not cl_paths.empty and "group" in cl_paths.columns:
                    cl_paths["path_label"] = cl_paths["from_var"] + "→" + cl_paths["to_var"]
                    pivot = cl_paths.pivot_table(
                        index="path_label", columns="group", values="beta", aggfunc="mean"
                    )
                    fig_hm = _go_hm.Figure(data=_go_hm.Heatmap(
                        z=pivot.values,
                        x=[str(c) for c in pivot.columns],
                        y=pivot.index.tolist(),
                        colorscale="RdBu",
                        zmid=0,
                        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                        texttemplate="%{text}",
                        hovertemplate="Path: %{y}<br>Group: %{x}<br>β=%{z:.3f}<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        title="Cross-Lagged Betas by Trajectory Group",
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif"),
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("No cross-lagged paths with group column found for heatmap.")
            except Exception as _e:
                st.info(f"Heatmap skipped: {_e}")

    elif section == "SHAP Feature Importance":
        if shap_df.empty:
            st.warning("Data file `figure3_real_shap.csv` not found in `data/` directory.")
        else:
            st.plotly_chart(shap_lollipop(shap_df), use_container_width=True)

    elif section == "ODE Intervention Simulation":
        if ode_df.empty:
            st.warning("Data file `figure3_real_ode.csv` not found in `data/` directory.")
        else:
            st.plotly_chart(ode_trajectory_chart(ode_df), use_container_width=True)

        # External validation — IPW risk difference forest plot
        if not ext_df.empty:
            st.markdown("#### External Validation — IPW Risk Difference")
            st.caption(
                "Forest plot of 5-year all-cause mortality risk difference (treated vs. control) "
                "across 9 external cohorts after inverse probability weighting (IPW). "
                "Negative RD = lower mortality in the intervention group. "
                "Diamond markers; error bars = 95% CI. Colour = balance adequacy after IPW."
            )
            st.plotly_chart(external_auc_chart(ext_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Counterfactual Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 What-if Intervention Simulation")

    if not st.session_state["prediction_done"]:
        st.info("Complete an analysis in **Tab 1** first.")
    else:
        result    = st.session_state["result"]
        orig_mace = result["mace_risk"]
        orig_comp = result["death_risk"]
        raw       = st.session_state["raw_inputs"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline MACE",        f"{orig_mace*100:.2f}%")
        c2.metric("Baseline Mortality",   f"{orig_comp*100:.2f}%")
        c3.metric("Follow-up",            f"{raw['follow_time']} months")
        st.markdown("---")

        # Intervention config (pure Python — no R needed)
        _CF_VARS = {
            "SBP":     {"label": "Systolic BP (mmHg)",   "min": 90,  "max": 200, "step": 1.0,  "key": "sbp"},
            "DBP":     {"label": "Diastolic BP (mmHg)",  "min": 50,  "max": 130, "step": 1.0,  "key": "dbp"},
            "BMI":     {"label": "BMI (kg/m²)",          "min": 15,  "max": 50,  "step": 0.5,  "key": "bmi"},
            "HTN":     {"label": "Hypertension (0/1)",   "min": 0,   "max": 1,   "step": 1.0,  "key": "htn"},
            "DM":      {"label": "Diabetes (0/1)",       "min": 0,   "max": 1,   "step": 1.0,  "key": "dm"},
            "Smoking": {"label": "Smoking (0/1)",        "min": 0,   "max": 1,   "step": 1.0,  "key": "smoking"},
        }

        col_ctrl, col_chart = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("**Select intervention variable:**")
            variable = st.selectbox(
                "Variable",
                list(_CF_VARS.keys()),
                format_func=lambda k: _CF_VARS[k]["label"],
                label_visibility="collapsed",
            )
            cfg = _CF_VARS[variable]

            # Current value from last raw input
            _key = cfg["key"]
            if _key in ("sbp", "dbp", "bmi"):
                _cur = raw[_key][-1]
            elif _key == "htn":
                _cur = float(int(raw["sbp"][-1] >= 140))
            else:
                _cur = 0.0

            target = st.slider(
                f"Target value ({cfg['label'].split('(')[1].rstrip(')')})",
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(_cur),
                step=float(cfg["step"]),
            )

            run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)

        with col_chart:
            if run_sim:
                with st.spinner("Running Python model with modified inputs..."):
                    try:
                        age_val    = st.session_state.get("age_input", 55)
                        gender_val = st.session_state.get("gender_input", "Male")

                        # Build counterfactual inputs
                        cf_sbp     = target if variable == "SBP"     else raw["sbp"][-1]
                        cf_dbp     = target if variable == "DBP"     else raw["dbp"][-1]
                        cf_bmi     = target if variable == "BMI"     else raw["bmi"][-1]
                        cf_htn     = int(target) if variable == "HTN"     else int(raw["sbp"][-1] >= 140)
                        cf_dm      = int(target) if variable == "DM"      else 0
                        cf_smoking = int(target) if variable == "Smoking" else 0

                        cf_result = py_predict(
                            age=age_val,
                            sex_male=(1 if gender_val == "Male" else 0),
                            bmi=cf_bmi,
                            sbp=cf_sbp,
                            dbp=cf_dbp,
                            htn=cf_htn,
                            dm=cf_dm,
                            smoking=cf_smoking,
                        )

                        if not cf_result.get("ok"):
                            st.error(f"Model error: {cf_result.get('error')}")
                        else:
                            cf_mace = cf_result["mace_risk"]
                            cf_comp = cf_result["death_risk"]

                            delta_mace = (cf_mace - orig_mace) * 100
                            delta_comp = (cf_comp - orig_comp) * 100

                            m1, m2 = st.columns(2)
                            m1.metric("MACE Risk Change",
                                      f"{cf_mace*100:.2f}%",
                                      delta=f"{delta_mace:+.2f}%",
                                      delta_color="inverse")
                            m2.metric("Mortality Risk Change",
                                      f"{cf_comp*100:.2f}%",
                                      delta=f"{delta_comp:+.2f}%",
                                      delta_color="inverse")

                            label = f"{variable} → {target}"
                            st.plotly_chart(
                                counterfactual_bar(orig_mace, orig_comp, cf_mace, cf_comp, label),
                                use_container_width=True,
                            )

                            if variable in ("SBP", "DBP") and delta_mace < 0:
                                st.info(
                                    "**Clinical Note:** Lowering blood pressure reduces MAP, "
                                    "which directly lowers the model's MACE and mortality estimates. "
                                    "This aligns with evidence from large RCTs (SPRINT, ACCORD) "
                                    "showing BP control reduces cardiovascular events in high-risk patients."
                                )
                    except Exception as e:
                        st.error(f"Simulation error: {e}")
            else:
                st.info("Adjust the slider and click **Run Simulation**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Multi-turn Clinical Dialogue
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💬 Multi-turn Clinical Dialogue")

    if not st.session_state["prediction_done"]:
        st.info("Complete an analysis in **Tab 1** first.")
    else:
        risk_level = st.session_state.get("risk_level", "—")
        comp_score = st.session_state.get("comp_score", 0)
        badge_cls  = {"Low": "risk-low", "Moderate": "risk-mid",
                      "High": "risk-high", "Very High": "risk-vhigh"}.get(risk_level, "risk-mid")
        st.markdown(
            f"<span class='risk-badge {badge_cls}'>{risk_level} Risk · {comp_score*100:.1f}%</span>"
            f"&nbsp; Ask follow-up questions about this patient's results, medications, or risk factors.",
            unsafe_allow_html=True,
        )
        st.markdown("")

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input(
            "Ask a clinical question..." if view_mode == "doctor"
            else "Ask about your health results..."
        ):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            system_prompt = (
                "You are a senior cardiologist assistant. "
                "Answer follow-up questions about the patient described below. "
                + ("Be specific, evidence-based, and concise."
                   if view_mode == "doctor" else
                   "Explain in simple, non-technical language suitable for a patient. "
                   "Avoid medical jargon. Be reassuring and clear.")
                + f"\n\n{st.session_state['system_context']}"
            )
            messages = [{"role": "system", "content": system_prompt}]
            messages += st.session_state["chat_history"]

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0.4,
                        )
                        reply = completion.choices[0].message.content
                        st.markdown(reply)
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as e:
                        st.error(f"LLM error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Health Education & Video
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📚 Health Education Resources")

    _index = load_index()

    if not st.session_state["prediction_done"]:
        st.info("Complete an analysis in **Tab 1** first to get personalised recommendations.")
        show_all = True
    else:
        show_all = False
        _rl  = st.session_state.get("risk_level", "Low")
        _es  = st.session_state.get("sbp_slope", 0.0)   # v13: sbp_slope replaces egfr_slope
        _ms  = st.session_state.get("map_slope", 0.0)
        recs = recommend_resources(_rl, _es, _ms, _index)

        if recs:
            st.markdown("#### Recommended for This Patient")
            render_resource_grid(recs, view_mode, key_prefix="rec_")
        else:
            st.info("No specific resources matched. Browse all resources below.")

    with st.expander("Browse All Resources", expanded=show_all):
        all_items = (
            [{**v, "type": "video"}  for v in _index.get("videos", [])] +
            [{**l, "type": "lottie"} for l in _index.get("lottie", [])]
        )
        if all_items:
            render_resource_grid(all_items, view_mode, key_prefix="all_")
        else:
            st.markdown("""
            <div style='background:#F8F9FA;border-radius:10px;padding:24px;text-align:center;
                        color:#7F8C8D;'>
              <div style='font-size:36px;margin-bottom:10px;'>📂</div>
              <div style='font-weight:600;'>No resources downloaded yet</div>
              <div style='font-size:13px;margin-top:8px;'>
                Add video URLs and Lottie URLs to <code>download_assets.py</code>,
                then run:<br/>
                <code style='background:#E8E8E8;padding:2px 8px;border-radius:4px;'>
                  python download_assets.py
                </code>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Literature & Bibliometrics
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📖 Literature & Bibliometrics")

    st.markdown(f"""
    <div class='biblio-card'>
      <div style='display:flex;gap:32px;flex-wrap:wrap;'>
        <div>
          <div style='font-size:11px;color:#5D6D7E;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;'>Total Papers</div>
          <div style='font-size:36px;font-weight:700;color:#1B3A57;'>{BIBLIO_STATS['total_papers']}</div>
        </div>
        <div>
          <div style='font-size:11px;color:#5D6D7E;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;'>Date Range</div>
          <div style='font-size:36px;font-weight:700;color:#1B3A57;'>{BIBLIO_STATS['date_range']}</div>
        </div>
        <div>
          <div style='font-size:11px;color:#5D6D7E;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;'>Top Journal</div>
          <div style='font-size:36px;font-weight:700;color:#1B3A57;'>{BIBLIO_STATS['top_journal']}</div>
        </div>
        <div style='flex:1;min-width:200px;'>
          <div style='font-size:11px;color:#5D6D7E;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;'>Top Keywords</div>
          <div style='font-size:13px;color:#2C3E50;margin-top:4px;line-height:1.8;'>
            {' · '.join(BIBLIO_STATS['keywords'][:5])}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    biblio_figs = bibliometrics_panel(BIBLIO_STATS)

    bib_tab_overview, bib_tab_trend, bib_tab_kw, bib_tab_journals = st.tabs(
        ["Overview", "Publication Trends", "Keywords", "Journals"]
    )

    with bib_tab_overview:
        st.markdown("### CKM Syndrome Literature Overview")
        st.markdown(f"""
        This bibliometric analysis covers **{BIBLIO_STATS['total_papers']} publications**
        from **{BIBLIO_STATS['date_range']}** indexed across major cardiovascular, renal,
        and metabolic medicine journals. The field has grown rapidly, with annual publication
        counts increasing from {BIBLIO_STATS['counts'][0]} in {BIBLIO_STATS['years'][0]}
        to {BIBLIO_STATS['counts'][-1]} in {BIBLIO_STATS['years'][-1]}.

        Key research themes include SGLT2 inhibitor cardio-renal protection, GBTM trajectory
        analysis, TyG index as a metabolic biomarker, and integrated CKM risk prediction models.
        """)
        _ov_c1, _ov_c2 = st.columns(2)
        with _ov_c1:
            st.plotly_chart(biblio_figs["trend"], use_container_width=True, key="bib_trend_ov")
        with _ov_c2:
            st.plotly_chart(biblio_figs["journals"], use_container_width=True, key="bib_journals_ov")

    with bib_tab_trend:
        st.plotly_chart(biblio_figs["trend"], use_container_width=True, key="bib_trend_tab")

    with bib_tab_kw:
        st.plotly_chart(biblio_figs["keywords"], use_container_width=True, key="bib_kw_tab")

    with bib_tab_journals:
        st.plotly_chart(biblio_figs["journals"], use_container_width=True, key="bib_journals_tab")

    st.markdown("---")
    st.markdown("### 综述图 — CKM Syndrome Pathobiologic Network")

    _REVIEW_SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" width="100%" height="400">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#4A6785"/>
        </marker>
        <marker id="arrow-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#A05060"/>
        </marker>
        <filter id="shadow">
          <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.15"/>
        </filter>
      </defs>
      <rect width="800" height="400" fill="#F8FAFF" rx="12"/>
      <ellipse cx="400" cy="200" rx="80" ry="40" fill="#1B3A57" filter="url(#shadow)"/>
      <text x="400" y="196" text-anchor="middle" fill="white" font-size="13" font-weight="bold" font-family="Inter,sans-serif">CKM</text>
      <text x="400" y="212" text-anchor="middle" fill="white" font-size="11" font-family="Inter,sans-serif">Syndrome</text>
      <ellipse cx="160" cy="120" rx="75" ry="35" fill="#A07830" filter="url(#shadow)"/>
      <text x="160" y="116" text-anchor="middle" fill="white" font-size="11" font-weight="bold" font-family="Inter,sans-serif">Metabolic</text>
      <text x="160" y="130" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">Dysregulation</text>
      <ellipse cx="160" cy="280" rx="75" ry="35" fill="#4E8B6F" filter="url(#shadow)"/>
      <text x="160" y="276" text-anchor="middle" fill="white" font-size="11" font-weight="bold" font-family="Inter,sans-serif">Cardiovascular</text>
      <text x="160" y="290" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">Burden</text>
      <ellipse cx="400" cy="60" rx="75" ry="35" fill="#6496B8" filter="url(#shadow)"/>
      <text x="400" y="56" text-anchor="middle" fill="white" font-size="11" font-weight="bold" font-family="Inter,sans-serif">Renal</text>
      <text x="400" y="70" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">Impairment</text>
      <ellipse cx="640" cy="120" rx="60" ry="32" fill="#A05060" filter="url(#shadow)"/>
      <text x="640" y="116" text-anchor="middle" fill="white" font-size="12" font-weight="bold" font-family="Inter,sans-serif">MACE</text>
      <text x="640" y="130" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">Events</text>
      <ellipse cx="640" cy="200" rx="60" ry="32" fill="#A05060" filter="url(#shadow)"/>
      <text x="640" y="196" text-anchor="middle" fill="white" font-size="11" font-weight="bold" font-family="Inter,sans-serif">CKD</text>
      <text x="640" y="210" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">Progression</text>
      <ellipse cx="640" cy="280" rx="60" ry="32" fill="#8B0000" filter="url(#shadow)"/>
      <text x="640" y="276" text-anchor="middle" fill="white" font-size="12" font-weight="bold" font-family="Inter,sans-serif">Mortality</text>
      <text x="640" y="290" text-anchor="middle" fill="white" font-size="10" font-family="Inter,sans-serif">All-cause</text>
      <line x1="322" y1="185" x2="235" y2="140" stroke="#4A6785" stroke-width="2" marker-end="url(#arrow)"/>
      <line x1="322" y1="215" x2="235" y2="265" stroke="#4A6785" stroke-width="2" marker-end="url(#arrow)"/>
      <line x1="360" y1="162" x2="400" y2="95" stroke="#4A6785" stroke-width="2" marker-end="url(#arrow)"/>
      <line x1="235" y1="115" x2="580" y2="120" stroke="#4A6785" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
      <line x1="235" y1="275" x2="580" y2="200" stroke="#4A6785" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
      <line x1="475" y1="65" x2="580" y2="120" stroke="#4A6785" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
      <line x1="475" y1="70" x2="580" y2="195" stroke="#4A6785" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
      <line x1="235" y1="285" x2="580" y2="275" stroke="#A05060" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow-red)"/>
      <text x="270" y="108" fill="#A07830" font-size="9" font-family="Inter,sans-serif">TyG&#8593; BMI&#8593;</text>
      <text x="255" y="260" fill="#4E8B6F" font-size="9" font-family="Inter,sans-serif">MAP&#8593; SBP&#8593;</text>
      <text x="370" y="108" fill="#6496B8" font-size="9" font-family="Inter,sans-serif">eGFR&#8595; Scr&#8593;</text>
      <text x="400" y="385" text-anchor="middle" fill="#5D6D7E" font-size="11" font-family="Inter,sans-serif">
        CKM Syndrome Pathobiologic Cascade — Metabolic · Cardiovascular · Renal → Outcomes
      </text>
    </svg>
    """

    components.html(_REVIEW_SVG, height=420)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Admin
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🔧 Admin Panel")
    if check_auth():
        section = st.radio(
            "Section",
            ["Upload Video", "Upload Literature", "Manage Videos"],
            horizontal=True,
            key="admin_section",
        )
        st.divider()
        if section == "Upload Video":
            render_video_upload()
        elif section == "Upload Literature":
            render_kb_upload()
        else:
            render_video_management()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.caption("📖 User Guide")
    with st.expander("Click to expand"):
        st.markdown("""
**Step 1 — Upload Health Reports (Optional)**
In Tab 1, upload 1–5 annual health check images or PDFs.
Assign a year label (Y1 = oldest, Y5 = most recent) to each file.
Click **Extract Data from Reports** and the system will auto-fill the biomarker fields using AI-powered OCR.

**Step 2 — Enter / Review Patient Data**
In Tab 1, verify the age, gender, and 5-year trajectory values.
Each field holds five comma-separated numbers (Year 1 → Year 5).
Edit any value manually if needed before running analysis.

**Step 3 — Run Intelligent Analysis**
Click **🚀 Start Intelligent Analysis**.
The system will classify trajectory groups, predict MACE and complication risks, and generate an AI clinical report.

**Step 4 — Explore Visual Analytics (Tab 2)**
Switch to **Tab 2** to explore trajectory phenotypes, CLPM causal paths, SHAP feature importance, and ODE intervention simulations.

**Step 5 — Explore What-If Simulations (Tab 3)**
Select a variable (e.g. SBP, BMI), drag the target slider, and click **▶ Run Simulation** to see predicted risk changes.

**Step 6 — Ask Follow-up Questions (Tab 4)**
Use the chat interface in **Tab 4** to ask clinical questions about the patient's results in natural language.

**Step 7 — Literature & Bibliometrics (Tab 6)**
Explore the CKM syndrome literature landscape, publication trends, keyword co-occurrence, and journal distribution.
        """)

    if st.button("▶ Replay Tutorial", key="replay_onboarding"):
        st.session_state["onboarding_complete"] = False
        st.session_state.pop("ob_step", None)
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.caption("⚙️ Interface Style")
    _mascot_now = st.session_state.get("show_mascot", True)
    _style_choice = st.radio(
        "Interface Style",
        ["🌸 Companion Mode", "📊 Minimal Mode"],
        index=0 if _mascot_now else 1,
        key="sidebar_style_radio",
        label_visibility="collapsed",
    )
    _new_mascot = (_style_choice == "🌸 Companion Mode")
    if _new_mascot != _mascot_now:
        st.session_state["show_mascot"] = _new_mascot
        st.rerun()

    st.divider()
    _story_btn_label = (
        "📖 Meet Your Health Guardians"
        if st.session_state.get("show_mascot", True)
        else "📄 CKM Pathobiologic Review"
    )
    if st.button(_story_btn_label, key="sidebar_story_btn"):
        st.session_state["show_story_modal"] = True
        st.session_state["story_anchor"] = None
        st.rerun()

