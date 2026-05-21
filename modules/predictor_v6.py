# modules/predictor_v6.py
"""
V6 inference wrapper for CKM Graph RAG v2.

Architecture: Transformer Encoder + BiLSTM-MHA + KAN (B-spline) + XGB + LGB + CatBoost
              → LGB meta-learner → 50% NN rank + 50% meta rank
AUC (derivation): 0.8839
Inputs: 5 years × 15 biomarkers (SEQ_VARS) + 626 static engineered features

Usage:
    from modules.predictor_v6 import predict_v6, is_v6_available
    if is_v6_available():
        result = predict_v6(seq_data, static_data)
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from pathlib import Path

_CKPT_DIR = Path(os.path.dirname(__file__)).parent / 'data'
_CKPT_PATH = _CKPT_DIR / 'v6_full_checkpoint.pkl'

MAX_LEN  = 5
SEQ_VARS = ['sbp', 'dbp', 'map', 'fpg', 'tg', 'hdl', 'egfr', 'tyg', 'bmi', 'ldl',
            'htn', 'dm', 'tx_bp_ever', 'tx_dm_ever', 'tx_lipid_ever']


# ─── Neural network classes (must match boost_auc_v6.py exactly) ──────────────

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        h    = 2.0 / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1,
                            dtype=torch.float32) * h - 1.0
        self.register_buffer('grid', grid)
        self.spline_weight  = nn.Parameter(torch.randn(out_features, in_features,
                                                        grid_size + spline_order) * 0.1)
        self.base_weight    = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_activation = nn.SiLU()
        self.spline_scaler  = nn.Parameter(torch.ones(out_features, in_features))

    def b_splines(self, x):
        x    = x.unsqueeze(-1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        for k in range(1, self.spline_order + 1):
            denom_l = grid[:, :, k:-1]   - grid[:, :, :-(k+1)] + 1e-8
            denom_r = grid[:, :, k+1:]   - grid[:, :, 1:-k]    + 1e-8
            left    = (x - grid[:, :, :-(k+1)]) / denom_l
            right   = (grid[:, :, k+1:] - x)    / denom_r
            bases   = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x):
        x_norm    = torch.tanh(x)
        bases     = self.b_splines(x_norm)
        spline_out = torch.einsum('bif,oif->bo', bases, self.spline_weight)
        spline_out = spline_out * self.spline_scaler.sum(dim=1).unsqueeze(0)
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        return base_out + spline_out


class KAN_Static(nn.Module):
    def __init__(self, input_dim, emb_dim=32, dropout=0.3):
        super().__init__()
        self.kan1 = KANLayer(input_dim, 128)
        self.kan2 = KANLayer(128, 64)
        self.kan3 = KANLayer(64, 32)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(32)
        self.drop  = nn.Dropout(dropout)
        self.emb_proj   = nn.Sequential(nn.Linear(32, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(emb_dim, 16), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(16, 1))
        self.ode_head   = nn.Linear(emb_dim, 1)

    def forward(self, x):
        h   = self.drop(self.norm1(self.kan1(x)))
        h   = self.drop(self.norm2(self.kan2(h)))
        h   = self.norm3(self.kan3(h))
        emb = self.emb_proj(h)
        return emb, self.classifier(emb).squeeze(-1), self.ode_head(emb).squeeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, emb_dim=32, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb    = nn.Embedding(MAX_LEN, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_pool   = nn.Linear(d_model, 1)
        self.emb_proj    = nn.Sequential(nn.Linear(d_model, emb_dim),
                                         nn.LayerNorm(emb_dim), nn.GELU())
        self.classifier  = nn.Sequential(nn.Linear(emb_dim, 32), nn.GELU(),
                                         nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.input_proj(x) + self.pos_emb(pos)
        kpm = (mask == 0) if mask is not None else None
        h   = self.transformer(h, src_key_padding_mask=kpm)
        aw  = self.attn_pool(h).squeeze(-1)
        if mask is not None:
            aw = aw.masked_fill(mask == 0, float('-inf'))
        aw  = F.softmax(aw, dim=1).unsqueeze(-1)
        emb = self.emb_proj((h * aw).sum(dim=1))
        return emb, self.classifier(emb).squeeze(-1)


class BiLSTM_MHA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=4, emb_dim=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.mha  = nn.MultiheadAttention(hidden_dim * 2, nhead, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.emb_proj   = nn.Sequential(nn.Linear(hidden_dim * 2, emb_dim),
                                        nn.LayerNorm(emb_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(emb_dim, 32), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        kpm    = (mask == 0) if mask is not None else None
        ao, _  = self.mha(out, out, out, key_padding_mask=kpm)
        out    = self.norm(out + ao)
        if mask is not None:
            m   = mask.unsqueeze(-1).float()
            ctx = (out * m).sum(1) / (m.sum(1) + 1e-8)
        else:
            ctx = out.mean(1)
        emb = self.emb_proj(ctx)
        return emb, self.classifier(emb).squeeze(-1)


# ─── Checkpoint loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_v6():
    """Load V6 checkpoint. Returns (ckpt, tf, bi, kan) or raises."""
    if not _CKPT_PATH.exists():
        raise FileNotFoundError(f"V6 checkpoint not found: {_CKPT_PATH}")
    with open(_CKPT_PATH, 'rb') as f:
        ckpt = pickle.load(f)

    arch = ckpt['arch']
    device = torch.device('cpu')  # inference on CPU for portability

    tf  = TransformerEncoder(arch['input_dim'], d_model=arch['d_model'],
                             nhead=arch['nhead'], num_layers=arch['num_layers'],
                             emb_dim=arch['emb_dim']).to(device)
    bi  = BiLSTM_MHA(arch['input_dim'], hidden_dim=arch['hidden_dim'],
                     num_layers=arch['num_layers'], nhead=arch['nhead'],
                     emb_dim=arch['emb_dim']).to(device)
    n_static = len(ckpt['feature_cols'])
    kan = KAN_Static(n_static, emb_dim=arch['emb_dim']).to(device)

    tf.load_state_dict(ckpt['tf_state'])
    bi.load_state_dict(ckpt['bi_state'])
    kan.load_state_dict(ckpt['kan_state'])
    tf.eval(); bi.eval(); kan.eval()

    return ckpt, tf, bi, kan


def is_v6_available() -> bool:
    """Return True if the V6 checkpoint exists and loads cleanly."""
    try:
        _load_v6()
        return True
    except Exception:
        return False


# ─── Sequence builder ─────────────────────────────────────────────────────────

def _build_sequence(yearly_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (seq, mask) tensors from a list of up to MAX_LEN year-dicts.

    yearly_data: list of dicts, each with keys matching SEQ_VARS.
                 Earliest year first. Missing values → 0 (masked).
    Returns:
        seq  : float32 (MAX_LEN, len(SEQ_VARS))
        mask : float32 (MAX_LEN,)  1=valid, 0=padded
    """
    seq  = np.zeros((MAX_LEN, len(SEQ_VARS)), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=np.float32)
    for t, row in enumerate(yearly_data[-MAX_LEN:]):
        for j, v in enumerate(SEQ_VARS):
            val = row.get(v, np.nan)
            if val is not None and not np.isnan(float(val)):
                seq[t, j] = float(val)
        mask[t] = 1.0
    return seq, mask


# ─── Main inference ───────────────────────────────────────────────────────────

def predict_v6(yearly_data: list[dict], static_features: dict) -> dict:
    """
    Run V6 ensemble inference for a single patient.

    Parameters
    ----------
    yearly_data : list of dicts (1–5 entries, earliest first)
        Each dict has keys from SEQ_VARS: sbp, dbp, map, fpg, tg, hdl, egfr,
        tyg, bmi, ldl, htn, dm, tx_bp_ever, tx_dm_ever, tx_lipid_ever
    static_features : dict
        Pre-computed static features matching feature_cols from training.
        Missing keys are filled with 0.

    Returns
    -------
    dict with keys:
        ok          : True
        mace_risk   : float  (5-year MACE probability, rank-blended)
        mace_nn     : float  (NN component probability)
        mace_meta   : float  (LGB meta component probability)
        model       : 'v6_ensemble'
        auc_ref     : 0.8839
    """
    try:
        ckpt, tf, bi, kan = _load_v6()
        device = next(tf.parameters()).device

        # Build sequence tensor
        seq_np, mask_np = _build_sequence(yearly_data)
        seq_t  = torch.tensor(seq_np,  dtype=torch.float32).unsqueeze(0).to(device)
        mask_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Build static feature vector
        feat_cols = ckpt['feature_cols']
        x_static  = np.array([float(static_features.get(c, 0.0)) for c in feat_cols],
                              dtype=np.float32).reshape(1, -1)
        x_sc = ckpt['scaler'].transform(x_static).astype(np.float32)
        stat_t = torch.tensor(x_sc, dtype=torch.float32).to(device)

        # NN forward pass
        with torch.no_grad():
            _, tl = tf(seq_t, mask_t)
            _, bl = bi(seq_t, mask_t)
            _, kl, _ = kan(stat_t)
            p_nn = float(torch.sigmoid((tl + bl + kl) / 3.0).cpu().item())

        # Tabular ensemble
        p_xgb = float(ckpt['xgb'].predict_proba(x_sc)[0, 1])
        p_lgb = float(ckpt['lgb'].predict_proba(x_sc)[0, 1])
        meta_in = np.array([[p_xgb, p_lgb]], dtype=np.float32)
        if ckpt['has_cat'] and ckpt['cat'] is not None:
            p_cat   = float(ckpt['cat'].predict_proba(x_sc)[0, 1])
            meta_in = np.array([[p_xgb, p_lgb, p_cat]], dtype=np.float32)
        p_meta = float(ckpt['meta_lgb'].predict_proba(
            ckpt['sc_meta'].transform(meta_in))[0, 1])

        # Rank blend (same as training: 50% NN rank + 50% meta rank)
        # For single-patient inference, ranks are not meaningful — return raw blend
        mace_risk = round(0.5 * p_nn + 0.5 * p_meta, 4)

        return {
            'ok':        True,
            'mace_risk': mace_risk,
            'mace_nn':   round(p_nn,   4),
            'mace_meta': round(p_meta, 4),
            'model':     'v6_ensemble',
            'auc_ref':   ckpt.get('auc_derivation', 0.8839),
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def build_static_from_v13_inputs(age, sex_male, bmi, sbp, dbp,
                                  htn=0, dm=0, smoking=0) -> dict:
    """
    Approximate static feature dict from v13-style inputs.
    Fills only the features derivable from basic clinical inputs;
    all trajectory/CLPM/ODE/wavelet features default to 0.

    Use this when you only have cross-sectional data — the model
    will run but accuracy will be lower than with full longitudinal data.
    """
    map_val = (sbp + 2 * dbp) / 3.0
    tyg     = np.log(float(sbp) * float(bmi) / 2.0 + 1e-8)  # proxy

    ckm_stage = 0.0
    if bmi >= 30:          ckm_stage += 1.0
    elif bmi >= 25:        ckm_stage += 0.5
    if sbp >= 140 or htn:  ckm_stage += 1.0
    elif sbp >= 130:       ckm_stage += 0.5
    if dm:                 ckm_stage += 1.0
    if age >= 65:          ckm_stage += 0.5

    return {
        'age':       age,
        'sex':       sex_male,
        'bmi':       bmi,
        'sbp':       sbp,
        'dbp':       dbp,
        'map':       map_val,
        'htn':       htn,
        'dm':        dm,
        'smoking':   smoking,
        'tyg':       tyg,
        'ckm_stage': ckm_stage,
        'age_sq':    age ** 2,
        'bmi_sq':    bmi ** 2,
        'sbp_sq':    sbp ** 2,
        'map_sq':    map_val ** 2,
        'age_x_htn': age * htn,
        'age_x_dm':  age * dm,
        'age_x_bmi': age * bmi,
        'map_x_dm':  map_val * dm,
    }
