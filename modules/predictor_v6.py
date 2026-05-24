# modules/predictor_v6.py
"""
V6 inference wrapper — pure-numpy implementation (no torch required).

Architecture: TransformerEncoder + BiLSTM-MHA + KAN (B-spline) + XGB + LGB + CatBoost
              -> LGB meta-learner -> 50% NN rank + 50% meta rank
AUC (derivation): 0.8839
Checkpoint format: numpy_compat_v1 (state dicts stored as numpy arrays)

Usage:
    from modules.predictor_v6 import predict_v6, is_v6_available
    if is_v6_available():
        result = predict_v6(seq_data, static_data)
"""
import os
import pickle
import numpy as np
from functools import lru_cache
from pathlib import Path

_CKPT_DIR  = Path(os.path.dirname(__file__)).parent / 'data'
_CKPT_PATH = _CKPT_DIR / 'v6_full_checkpoint.pkl'

MAX_LEN  = 5
SEQ_VARS = ['sbp', 'dbp', 'map', 'fpg', 'tg', 'hdl', 'egfr', 'tyg', 'bmi', 'ldl',
            'htn', 'dm', 'tx_bp_ever', 'tx_dm_ever', 'tx_lipid_ever']


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _silu(x):
    return x / (1.0 + np.exp(-x))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias

def _linear(x, weight, bias):
    # weight: (out, in), x: (..., in)
    return x @ weight.T + bias

def _dropout_identity(x):
    # Inference — dropout is identity
    return x


# ---------------------------------------------------------------------------
# KAN Layer (B-spline)
# ---------------------------------------------------------------------------

class _KANLayer:
    def __init__(self, w):
        self.grid         = w['grid']            # (n_grid_pts,)
        self.spline_weight = w['spline_weight']  # (out, in, grid_size+order)
        self.base_weight   = w['base_weight']    # (out, in)
        self.spline_scaler = w['spline_scaler']  # (out, in)

    def _b_splines(self, x, order=3):
        # x: (batch, in_features)  -> returns (batch, in_features, n_bases)
        x    = x[:, :, np.newaxis]           # (B, in, 1)
        grid = self.grid[np.newaxis, np.newaxis, :]  # (1, 1, G)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).astype(np.float32)
        for k in range(1, order + 1):
            denom_l = grid[:, :, k:-1]   - grid[:, :, :-(k+1)] + 1e-8
            denom_r = grid[:, :, k+1:]   - grid[:, :, 1:-k]    + 1e-8
            left    = (x - grid[:, :, :-(k+1)]) / denom_l
            right   = (grid[:, :, k+1:] - x)    / denom_r
            bases   = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases  # (B, in, n_bases)

    def forward(self, x):
        # x: (B, in_features)
        x_norm     = np.tanh(x)
        bases      = self._b_splines(x_norm)                        # (B, in, n_bases)
        # spline_out: einsum('bif,oif->bo', bases, spline_weight)
        spline_out = np.einsum('bif,oif->bo', bases, self.spline_weight)
        scaler_sum = self.spline_scaler.sum(axis=1)                  # (out,)
        spline_out = spline_out * scaler_sum[np.newaxis, :]
        base_out   = _linear(_silu(x), self.base_weight)
        return base_out + spline_out


# ---------------------------------------------------------------------------
# KAN_Static
# ---------------------------------------------------------------------------

class _KAN_Static:
    def __init__(self, ws):
        self.kan1 = _KANLayer(ws['kan1'])
        self.kan2 = _KANLayer(ws['kan2'])
        self.kan3 = _KANLayer(ws['kan3'])
        self.norm1_w = ws['norm1_w']; self.norm1_b = ws['norm1_b']
        self.norm2_w = ws['norm2_w']; self.norm2_b = ws['norm2_b']
        self.norm3_w = ws['norm3_w']; self.norm3_b = ws['norm3_b']
        # emb_proj: Linear + LayerNorm + GELU
        self.ep_lin_w = ws['ep_lin_w']; self.ep_lin_b = ws['ep_lin_b']
        self.ep_ln_w  = ws['ep_ln_w'];  self.ep_ln_b  = ws['ep_ln_b']
        # classifier: Linear + GELU + Dropout + Linear
        self.cl_lin1_w = ws['cl_lin1_w']; self.cl_lin1_b = ws['cl_lin1_b']
        self.cl_lin2_w = ws['cl_lin2_w']; self.cl_lin2_b = ws['cl_lin2_b']

    def forward(self, x):
        h   = _layer_norm(self.kan1.forward(x), self.norm1_w, self.norm1_b)
        h   = _layer_norm(self.kan2.forward(h), self.norm2_w, self.norm2_b)
        h   = _layer_norm(self.kan3.forward(h), self.norm3_w, self.norm3_b)
        emb = _gelu(_layer_norm(_linear(h, self.ep_lin_w, self.ep_lin_b),
                                self.ep_ln_w, self.ep_ln_b))
        logit = _linear(_gelu(_linear(emb, self.cl_lin1_w, self.cl_lin1_b)),
                        self.cl_lin2_w, self.cl_lin2_b)
        return emb, logit.squeeze(-1)


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------

class _TransformerEncoder:
    def __init__(self, ws, d_model, nhead, num_layers):
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_layers = num_layers

        self.input_proj_w = ws['input_proj_w']  # (d_model, input_dim)
        self.input_proj_b = ws['input_proj_b']
        self.pos_emb      = ws['pos_emb']        # (MAX_LEN, d_model)

        # Per-layer weights (list of dicts)
        self.layers = ws['layers']

        # Attention pooling
        self.attn_pool_w = ws['attn_pool_w']  # (1, d_model)
        self.attn_pool_b = ws['attn_pool_b']

        # emb_proj: Linear + LayerNorm + GELU
        self.ep_lin_w = ws['ep_lin_w']; self.ep_lin_b = ws['ep_lin_b']
        self.ep_ln_w  = ws['ep_ln_w'];  self.ep_ln_b  = ws['ep_ln_b']

        # classifier
        self.cl_lin1_w = ws['cl_lin1_w']; self.cl_lin1_b = ws['cl_lin1_b']
        self.cl_lin2_w = ws['cl_lin2_w']; self.cl_lin2_b = ws['cl_lin2_b']

    def _mha(self, x, lw, mask=None):
        # x: (B, T, d_model), single-head for simplicity merged across heads
        B, T, D = x.shape
        H = self.nhead
        head_dim = D // H

        # in_proj: combined QKV weight (3*D, D)
        qkv = x @ lw['in_proj_w'].T + lw['in_proj_b']  # (B, T, 3D)
        q, k, v = np.split(qkv, 3, axis=-1)             # each (B, T, D)

        # Reshape to (B, H, T, head_dim)
        q = q.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)

        scale  = np.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) with 0=pad -> fill pad positions with -inf
            pad = (mask == 0)[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, T)
            scores = np.where(pad, -1e9, scores)

        attn = _softmax(scores, axis=-1)                    # (B, H, T, T)
        ctx  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)

        # out_proj
        out = ctx @ lw['out_proj_w'].T + lw['out_proj_b']
        return out

    def _ffn(self, x, lw):
        h = _gelu(_linear(x, lw['ffn_lin1_w'], lw['ffn_lin1_b']))
        return _linear(h, lw['ffn_lin2_w'], lw['ffn_lin2_b'])

    def forward(self, x, mask=None):
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        pos = np.arange(T)
        h   = x @ self.input_proj_w.T + self.input_proj_b  # (B, T, d_model)
        h   = h + self.pos_emb[np.newaxis, :T, :]

        for lw in self.layers:
            # Self-attention sublayer
            sa_out = self._mha(h, lw, mask)
            h2 = _layer_norm(h + sa_out, lw['norm1_w'], lw['norm1_b'])
            # FFN sublayer
            ff_out = self._ffn(h2, lw)
            h = _layer_norm(h2 + ff_out, lw['norm2_w'], lw['norm2_b'])

        # Attention pooling
        aw = (h @ self.attn_pool_w.T + self.attn_pool_b).squeeze(-1)  # (B, T)
        if mask is not None:
            aw = np.where(mask == 0, -1e9, aw)
        aw  = _softmax(aw, axis=1)[:, :, np.newaxis]                   # (B, T, 1)
        ctx = (h * aw).sum(axis=1)                                     # (B, d_model)

        emb   = _gelu(_layer_norm(_linear(ctx, self.ep_lin_w, self.ep_lin_b),
                                  self.ep_ln_w, self.ep_ln_b))
        logit = _linear(_gelu(_linear(emb, self.cl_lin1_w, self.cl_lin1_b)),
                        self.cl_lin2_w, self.cl_lin2_b)
        return emb, logit.squeeze(-1)


# ---------------------------------------------------------------------------
# BiLSTM_MHA
# ---------------------------------------------------------------------------

class _BiLSTM_MHA:
    def __init__(self, ws, hidden_dim, nhead):
        self.hidden_dim = hidden_dim
        self.nhead      = nhead
        # LSTM weights stored per layer/direction
        self.lstm_layers = ws['lstm_layers']  # list of (fwd_w, bwd_w) dicts
        # MHA
        self.mha_w  = ws['mha_w']
        self.norm_w = ws['norm_w']; self.norm_b = ws['norm_b']
        # emb_proj
        self.ep_lin_w = ws['ep_lin_w']; self.ep_lin_b = ws['ep_lin_b']
        self.ep_ln_w  = ws['ep_ln_w'];  self.ep_ln_b  = ws['ep_ln_b']
        # classifier
        self.cl_lin1_w = ws['cl_lin1_w']; self.cl_lin1_b = ws['cl_lin1_b']
        self.cl_lin2_w = ws['cl_lin2_w']; self.cl_lin2_b = ws['cl_lin2_b']

    def _lstm_cell(self, x_t, h, c, weight_ih, weight_hh, bias_ih, bias_hh):
        # x_t: (B, input), h/c: (B, hidden)
        gates = x_t @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
        i, f, g, o = np.split(gates, 4, axis=-1)
        i = _sigmoid(i); f = _sigmoid(f); g = np.tanh(g); o = _sigmoid(o)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    def _run_lstm(self, x, layer_w):
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        H = self.hidden_dim

        # Forward
        h_f = np.zeros((B, H), dtype=np.float32)
        c_f = np.zeros((B, H), dtype=np.float32)
        fwd_out = []
        for t in range(T):
            h_f, c_f = self._lstm_cell(x[:, t, :], h_f, c_f,
                                        layer_w['weight_ih_l0'],
                                        layer_w['weight_hh_l0'],
                                        layer_w['bias_ih_l0'],
                                        layer_w['bias_hh_l0'])
            fwd_out.append(h_f)

        # Backward
        h_b = np.zeros((B, H), dtype=np.float32)
        c_b = np.zeros((B, H), dtype=np.float32)
        bwd_out = []
        for t in range(T - 1, -1, -1):
            h_b, c_b = self._lstm_cell(x[:, t, :], h_b, c_b,
                                        layer_w['weight_ih_l0_reverse'],
                                        layer_w['weight_hh_l0_reverse'],
                                        layer_w['bias_ih_l0_reverse'],
                                        layer_w['bias_hh_l0_reverse'])
            bwd_out.append(h_b)
        bwd_out = bwd_out[::-1]

        out = np.concatenate([np.stack(fwd_out, axis=1),
                               np.stack(bwd_out, axis=1)], axis=-1)  # (B, T, 2H)
        return out

    def _mha(self, x, mask=None):
        B, T, D = x.shape
        H       = self.nhead
        head_dim = D // H
        lw      = self.mha_w

        qkv = x @ lw['in_proj_w'].T + lw['in_proj_b']
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)

        scale  = np.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            pad = (mask == 0)[:, np.newaxis, np.newaxis, :]
            scores = np.where(pad, -1e9, scores)
        attn = _softmax(scores, axis=-1)
        ctx  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return ctx @ lw['out_proj_w'].T + lw['out_proj_b']

    def forward(self, x, mask=None):
        # Run each LSTM layer, feeding output of previous as input
        out = x
        for layer_w in self.lstm_layers:
            out = self._run_lstm(out, layer_w)

        ao  = self._mha(out, mask)
        out = _layer_norm(out + ao, self.norm_w, self.norm_b)

        if mask is not None:
            m   = mask[:, :, np.newaxis].astype(np.float32)
            ctx = (out * m).sum(axis=1) / (m.sum(axis=1) + 1e-8)
        else:
            ctx = out.mean(axis=1)

        emb   = _gelu(_layer_norm(_linear(ctx, self.ep_lin_w, self.ep_lin_b),
                                  self.ep_ln_w, self.ep_ln_b))
        logit = _linear(_gelu(_linear(emb, self.cl_lin1_w, self.cl_lin1_b)),
                        self.cl_lin2_w, self.cl_lin2_b)
        return emb, logit.squeeze(-1)


# ---------------------------------------------------------------------------
# Checkpoint parsing helpers
# ---------------------------------------------------------------------------

def _g(sd, key):
    """Get numpy array from state dict."""
    v = sd[key]
    if hasattr(v, 'numpy'):
        return v.numpy()
    return np.asarray(v, dtype=np.float32)


def _parse_kan_weights(sd, prefix):
    """Extract KANLayer weights from a flat state dict."""
    return {
        'grid':          _g(sd, f'{prefix}.grid'),
        'spline_weight': _g(sd, f'{prefix}.spline_weight'),
        'base_weight':   _g(sd, f'{prefix}.base_weight'),
        'spline_scaler': _g(sd, f'{prefix}.spline_scaler'),
    }


def _parse_kan_static(sd):
    ws = {
        'kan1': _parse_kan_weights(sd, 'kan1'),
        'kan2': _parse_kan_weights(sd, 'kan2'),
        'kan3': _parse_kan_weights(sd, 'kan3'),
        'norm1_w': _g(sd, 'norm1.weight'), 'norm1_b': _g(sd, 'norm1.bias'),
        'norm2_w': _g(sd, 'norm2.weight'), 'norm2_b': _g(sd, 'norm2.bias'),
        'norm3_w': _g(sd, 'norm3.weight'), 'norm3_b': _g(sd, 'norm3.bias'),
        'ep_lin_w': _g(sd, 'emb_proj.0.weight'), 'ep_lin_b': _g(sd, 'emb_proj.0.bias'),
        'ep_ln_w':  _g(sd, 'emb_proj.1.weight'), 'ep_ln_b':  _g(sd, 'emb_proj.1.bias'),
        'cl_lin1_w': _g(sd, 'classifier.0.weight'), 'cl_lin1_b': _g(sd, 'classifier.0.bias'),
        'cl_lin2_w': _g(sd, 'classifier.3.weight'), 'cl_lin2_b': _g(sd, 'classifier.3.bias'),
    }
    return ws


def _parse_transformer(sd, num_layers):
    layers = []
    for i in range(num_layers):
        p = f'transformer.layers.{i}'
        lw = {
            'in_proj_w':  _g(sd, f'{p}.self_attn.in_proj_weight'),
            'in_proj_b':  _g(sd, f'{p}.self_attn.in_proj_bias'),
            'out_proj_w': _g(sd, f'{p}.self_attn.out_proj.weight'),
            'out_proj_b': _g(sd, f'{p}.self_attn.out_proj.bias'),
            'norm1_w': _g(sd, f'{p}.norm1.weight'), 'norm1_b': _g(sd, f'{p}.norm1.bias'),
            'norm2_w': _g(sd, f'{p}.norm2.weight'), 'norm2_b': _g(sd, f'{p}.norm2.bias'),
            'ffn_lin1_w': _g(sd, f'{p}.linear1.weight'), 'ffn_lin1_b': _g(sd, f'{p}.linear1.bias'),
            'ffn_lin2_w': _g(sd, f'{p}.linear2.weight'), 'ffn_lin2_b': _g(sd, f'{p}.linear2.bias'),
        }
        layers.append(lw)

    ws = {
        'input_proj_w': _g(sd, 'input_proj.weight'),
        'input_proj_b': _g(sd, 'input_proj.bias'),
        'pos_emb':      _g(sd, 'pos_emb.weight'),
        'layers':       layers,
        'attn_pool_w':  _g(sd, 'attn_pool.weight'),
        'attn_pool_b':  _g(sd, 'attn_pool.bias'),
        'ep_lin_w': _g(sd, 'emb_proj.0.weight'), 'ep_lin_b': _g(sd, 'emb_proj.0.bias'),
        'ep_ln_w':  _g(sd, 'emb_proj.1.weight'), 'ep_ln_b':  _g(sd, 'emb_proj.1.bias'),
        'cl_lin1_w': _g(sd, 'classifier.0.weight'), 'cl_lin1_b': _g(sd, 'classifier.0.bias'),
        'cl_lin2_w': _g(sd, 'classifier.3.weight'), 'cl_lin2_b': _g(sd, 'classifier.3.bias'),
    }
    return ws


def _parse_bilstm(sd, num_layers):
    lstm_layers = []
    for i in range(num_layers):
        if i == 0:
            lw = {
                'weight_ih_l0':         _g(sd, 'lstm.weight_ih_l0'),
                'weight_hh_l0':         _g(sd, 'lstm.weight_hh_l0'),
                'bias_ih_l0':           _g(sd, 'lstm.bias_ih_l0'),
                'bias_hh_l0':           _g(sd, 'lstm.bias_hh_l0'),
                'weight_ih_l0_reverse': _g(sd, 'lstm.weight_ih_l0_reverse'),
                'weight_hh_l0_reverse': _g(sd, 'lstm.weight_hh_l0_reverse'),
                'bias_ih_l0_reverse':   _g(sd, 'lstm.bias_ih_l0_reverse'),
                'bias_hh_l0_reverse':   _g(sd, 'lstm.bias_hh_l0_reverse'),
            }
        else:
            lw = {
                'weight_ih_l0':         _g(sd, f'lstm.weight_ih_l{i}'),
                'weight_hh_l0':         _g(sd, f'lstm.weight_hh_l{i}'),
                'bias_ih_l0':           _g(sd, f'lstm.bias_ih_l{i}'),
                'bias_hh_l0':           _g(sd, f'lstm.bias_hh_l{i}'),
                'weight_ih_l0_reverse': _g(sd, f'lstm.weight_ih_l{i}_reverse'),
                'weight_hh_l0_reverse': _g(sd, f'lstm.weight_hh_l{i}_reverse'),
                'bias_ih_l0_reverse':   _g(sd, f'lstm.bias_ih_l{i}_reverse'),
                'bias_hh_l0_reverse':   _g(sd, f'lstm.bias_hh_l{i}_reverse'),
            }
        lstm_layers.append(lw)

    ws = {
        'lstm_layers': lstm_layers,
        'mha_w': {
            'in_proj_w':  _g(sd, 'mha.in_proj_weight'),
            'in_proj_b':  _g(sd, 'mha.in_proj_bias'),
            'out_proj_w': _g(sd, 'mha.out_proj.weight'),
            'out_proj_b': _g(sd, 'mha.out_proj.bias'),
        },
        'norm_w': _g(sd, 'norm.weight'), 'norm_b': _g(sd, 'norm.bias'),
        'ep_lin_w': _g(sd, 'emb_proj.0.weight'), 'ep_lin_b': _g(sd, 'emb_proj.0.bias'),
        'ep_ln_w':  _g(sd, 'emb_proj.1.weight'), 'ep_ln_b':  _g(sd, 'emb_proj.1.bias'),
        'cl_lin1_w': _g(sd, 'classifier.0.weight'), 'cl_lin1_b': _g(sd, 'classifier.0.bias'),
        'cl_lin2_w': _g(sd, 'classifier.3.weight'), 'cl_lin2_b': _g(sd, 'classifier.3.bias'),
    }
    return ws


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_v6():
    if not _CKPT_PATH.exists():
        raise FileNotFoundError(f"V6 checkpoint not found: {_CKPT_PATH}")
    with open(_CKPT_PATH, 'rb') as f:
        ckpt = pickle.load(f)

    arch = ckpt['arch']
    fmt  = ckpt.get('format', 'torch_native')

    if fmt == 'numpy_compat_v1':
        tf_sd  = ckpt['tf_state_np']
        bi_sd  = ckpt['bi_state_np']
        kan_sd = ckpt['kan_state_np']
    else:
        # Try to extract numpy arrays from torch state_dicts
        def _to_np(sd):
            return {k: (v.numpy() if hasattr(v, 'numpy') else np.asarray(v))
                    for k, v in sd.items()}
        tf_sd  = _to_np(ckpt['tf_state'])
        bi_sd  = _to_np(ckpt['bi_state'])
        kan_sd = _to_np(ckpt['kan_state'])

    tf  = _TransformerEncoder(_parse_transformer(tf_sd, arch['num_layers']),
                              arch['d_model'], arch['nhead'], arch['num_layers'])
    bi  = _BiLSTM_MHA(_parse_bilstm(bi_sd, arch['num_layers']),
                      arch['hidden_dim'], arch['nhead'])
    kan = _KAN_Static(_parse_kan_static(kan_sd))

    return ckpt, tf, bi, kan


def is_v6_available() -> bool:
    try:
        _load_v6()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _build_sequence(yearly_data: list) -> tuple:
    seq  = np.zeros((MAX_LEN, len(SEQ_VARS)), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=np.float32)
    for t, row in enumerate(yearly_data[-MAX_LEN:]):
        for j, v in enumerate(SEQ_VARS):
            val = row.get(v, np.nan)
            if val is not None and not np.isnan(float(val)):
                seq[t, j] = float(val)
        mask[t] = 1.0
    return seq, mask


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def predict_v6(yearly_data: list, static_features: dict) -> dict:
    """
    Run V6 ensemble inference for a single patient.

    Parameters
    ----------
    yearly_data : list of dicts (1-5 entries, earliest first)
    static_features : dict of pre-computed static features

    Returns
    -------
    dict with keys: ok, mace_risk, mace_nn, mace_meta, model, auc_ref
    """
    try:
        ckpt, tf, bi, kan = _load_v6()

        seq_np, mask_np = _build_sequence(yearly_data)
        seq_in  = seq_np[np.newaxis, :, :]    # (1, T, 15)
        mask_in = mask_np[np.newaxis, :]       # (1, T)

        feat_cols = ckpt['feature_cols']
        x_static  = np.array([float(static_features.get(c, 0.0)) for c in feat_cols],
                              dtype=np.float32).reshape(1, -1)
        x_sc = ckpt['scaler'].transform(x_static).astype(np.float32)

        _, tl = tf.forward(seq_in, mask_in)
        _, bl = bi.forward(seq_in, mask_in)
        _, kl = kan.forward(x_sc)

        p_nn = float(_sigmoid((tl + bl + kl) / 3.0).item()
                     if hasattr((tl + bl + kl) / 3.0, 'item')
                     else _sigmoid(float((tl + bl + kl) / 3.0)))

        p_xgb = float(ckpt['xgb'].predict_proba(x_sc)[0, 1])
        p_lgb = float(ckpt['lgb'].predict_proba(x_sc)[0, 1])
        meta_in = np.array([[p_xgb, p_lgb]], dtype=np.float32)
        if ckpt.get('has_cat') and ckpt.get('cat') is not None:
            p_cat   = float(ckpt['cat'].predict_proba(x_sc)[0, 1])
            meta_in = np.array([[p_xgb, p_lgb, p_cat]], dtype=np.float32)
        p_meta = float(ckpt['meta_lgb'].predict_proba(
            ckpt['sc_meta'].transform(meta_in))[0, 1])

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
    map_val = (sbp + 2 * dbp) / 3.0
    tyg     = np.log(float(sbp) * float(bmi) / 2.0 + 1e-8)

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
