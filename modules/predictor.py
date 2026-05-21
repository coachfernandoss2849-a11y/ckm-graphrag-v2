# modules/predictor.py
"""
Pure-Python prediction engine for CKM Graph RAG v2.
Replaces the R/Plumber backend entirely.

Models: ev_v13 ensemble (LightGBM + XGBoost + LogisticRegression)
Features: 19 engineered features derived from basic clinical inputs
Outcomes: 5-year all-cause mortality (death) + MACE
"""
import os
import pickle
import numpy as np
import pandas as pd
from functools import lru_cache

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@lru_cache(maxsize=1)
def _load_models():
    death_path = os.path.join(_MODEL_DIR, 'model_death.pkl')
    mace_path  = os.path.join(_MODEL_DIR, 'model_mace.pkl')
    with open(death_path, 'rb') as f:
        death_ckpt = pickle.load(f)
    with open(mace_path, 'rb') as f:
        mace_ckpt = pickle.load(f)
    return death_ckpt, mace_ckpt


def _build_features(age: float, sex_male: int, bmi: float,
                    sbp: float, dbp: float,
                    htn: int, dm: int, smoking: int) -> pd.DataFrame:
    """
    Build the 19-feature DataFrame expected by the v13 ensemble.
    Derived features are computed here — no R needed.
    """
    map_val = (sbp + 2 * dbp) / 3.0

    # CKM approximate staging score (0-4 proxy)
    ckm_approx = 0.0
    if bmi >= 30:          ckm_approx += 1.0
    elif bmi >= 25:        ckm_approx += 0.5
    if sbp >= 140 or htn:  ckm_approx += 1.0
    elif sbp >= 130:       ckm_approx += 0.5
    if dm:                 ckm_approx += 1.0
    if age >= 65:          ckm_approx += 0.5

    row = {
        'age':        age,
        'sex_male':   sex_male,
        'bmi':        bmi,
        'sbp':        sbp,
        'dbp':        dbp,
        'map':        map_val,
        'htn':        htn,
        'dm':         dm,
        'smoking':    smoking,
        'age_sq':     age ** 2,
        'bmi_sq':     bmi ** 2,
        'sbp_sq':     sbp ** 2,
        'map_sq':     map_val ** 2,
        'age_x_htn':  age * htn,
        'age_x_dm':   age * dm,
        'age_x_bmi':  age * bmi,
        'map_x_dm':   map_val * dm,
        'ckm_approx': ckm_approx,
        'ckm_x_age':  ckm_approx * age,
    }
    return pd.DataFrame([row])


def _ensemble_predict(ckpt: dict, X: pd.DataFrame) -> float:
    """Run LGB + XGB + LR ensemble, return mean probability."""
    feat_cols = ckpt['feature_cols']
    scaler    = ckpt['scaler']
    X_ordered = X[feat_cols]
    X_scaled  = scaler.transform(X_ordered)

    lgb_p = float(ckpt['lgb'].predict_proba(X_scaled)[0, 1])
    xgb_p = float(ckpt['xgb'].predict_proba(X_scaled)[0, 1])
    lr_p  = float(ckpt['lr'].predict_proba(X_scaled)[0, 1])
    return round((lgb_p + xgb_p + lr_p) / 3.0, 4)


def predict(
    age: float,
    sex_male: int,
    bmi: float,
    sbp: float,
    dbp: float,
    htn: int = 0,
    dm: int = 0,
    smoking: int = 0,
) -> dict:
    """
    Main prediction entry point.

    Parameters
    ----------
    age       : years
    sex_male  : 1=male, 0=female
    bmi       : kg/m²
    sbp       : systolic BP mmHg
    dbp       : diastolic BP mmHg
    htn       : hypertension diagnosis (0/1)
    dm        : diabetes diagnosis (0/1)
    smoking   : current smoker (0/1)

    Returns
    -------
    dict with keys:
        ok          : True
        death_risk  : float  (5-year all-cause mortality probability)
        mace_risk   : float  (5-year MACE probability)
        map         : float  (computed MAP)
        ckm_approx  : float  (approximate CKM stage 0-4)
        model       : 'v13_ensemble'
    """
    try:
        death_ckpt, mace_ckpt = _load_models()
        X = _build_features(age, sex_male, bmi, sbp, dbp, htn, dm, smoking)

        death_risk = _ensemble_predict(death_ckpt, X)
        mace_risk  = _ensemble_predict(mace_ckpt,  X)
        map_val    = float(X['map'].iloc[0])
        ckm_approx = float(X['ckm_approx'].iloc[0])

        return {
            'ok':         True,
            'death_risk': death_risk,
            'mace_risk':  mace_risk,
            'map':        round(map_val, 1),
            'ckm_approx': ckm_approx,
            'model':      'v13_ensemble_lgb_xgb_lr',
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def risk_level(p: float) -> str:
    """Map probability to risk tier."""
    if p < 0.05:   return 'Low'
    if p < 0.10:   return 'Moderate'
    if p < 0.20:   return 'High'
    return 'Very High'
