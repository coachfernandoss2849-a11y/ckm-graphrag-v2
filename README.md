# CKM Graph RAG v2 — Intelligent Clinical Decision Support

A comprehensive clinical decision support system for CKM (Cardiovascular-Kidney-Metabolic) syndrome.

## Features

| Module | Description |
|--------|-------------|
| 📋 Patient Input & Prediction | OCR auto-fill, 5-year trajectory, GBTM+Cox risk prediction |
| 📊 Visual Analytics Dashboard | Trajectory phenotypes, CLPM causal paths, SHAP importance, ODE simulation |
| 🔬 Counterfactual Simulation | What-if intervention analysis |
| 💬 Clinical Dialogue | Multi-turn RAG-powered clinical chat |
| 📚 Health Education & Video | Personalised video resources |
| 📖 Literature & Bibliometrics | Publication trends, keyword analysis, review network |
| 🔧 Admin | Video/KB upload, ChromaDB management |

## Quick Start (Local)

```bash
cd D:/C/CKM_GraphRAG_v2
pip install -r requirements.txt
python launch.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Add secrets in Streamlit Cloud dashboard:
   ```toml
   ZHIPUAI_API_KEY = "your_key_here"
   R_API_URL = "https://your-r-backend.com/predict"
   ```

## Architecture

```
CKM Graph RAG v2
├── app.py                    # Main Streamlit app (7 tabs)
├── modules/
│   ├── viz.py               # 14 Plotly chart functions
│   ├── counterfactual.py    # What-if simulation engine
│   ├── ocr.py               # ZhipuAI Vision OCR
│   ├── science.py           # Health education resources
│   └── admin.py             # Admin panel
├── data/                    # Panel data CSVs
├── assets/
│   ├── ip-states/           # 6 IP guardian mascot images
│   └── subtitles/           # VTT subtitle files
├── chroma_db/               # Vector database
└── .streamlit/config.toml   # Streamlit configuration
```

## Data Sources

- GBTM trajectory model: Sanming cohort N=95,240
- CLPM causal paths: TyG→MAP→eGFR cascade
- SHAP importance: XGBoost+LightGBM+CatBoost ensemble
- ODE simulation: 5 intervention scenarios
- External validation: 9 international cohorts
