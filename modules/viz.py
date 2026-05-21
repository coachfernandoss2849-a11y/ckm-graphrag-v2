"""
viz.py — Enhanced visualization module for CKM GraphRAG v2 Streamlit agent.
All charts use Plotly with transparent backgrounds and the CKM palette.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY  = '#1B3A57'
SLATE = '#4A6785'
SKY   = '#6496B8'
SAGE  = '#4E8B6F'
ROSE  = '#A05060'
OCHRE = '#A07830'
STEEL = '#8090A0'
MIST  = '#B8C8D8'

_FONT = 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
_BG   = 'rgba(0,0,0,0)'

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(family=_FONT, color=NAVY),
    margin=dict(l=40, r=40, t=50, b=40),
)


def _apply_base(fig, title='', height=None):
    kw = dict(**_BASE_LAYOUT, title=dict(text=title, font=dict(size=15, color=NAVY)))
    if height:
        kw['height'] = height
    fig.update_layout(**kw)
    return fig


# ---------------------------------------------------------------------------
# 1. risk_gauge
# ---------------------------------------------------------------------------
def risk_gauge(mace: float, comp: float) -> go.Figure:
    """Dual gauge chart for MACE and complication risk (0-1 scale)."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['MACE Risk', 'Complication Risk'],
    )

    def _gauge(value, color):
        return go.Indicator(
            mode='gauge+number',
            value=round(value * 100, 1),
            number=dict(suffix='%', font=dict(size=28, color=color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=10)),
                bar=dict(color=color, thickness=0.6),
                bgcolor=_BG,
                steps=[
                    dict(range=[0, 30],  color='rgba(78,139,111,0.15)'),
                    dict(range=[30, 60], color='rgba(160,120,48,0.15)'),
                    dict(range=[60, 100], color='rgba(160,80,96,0.15)'),
                ],
                threshold=dict(
                    line=dict(color=NAVY, width=2),
                    thickness=0.75,
                    value=round(value * 100, 1),
                ),
            ),
        )

    fig.add_trace(_gauge(mace, ROSE), row=1, col=1)
    fig.add_trace(_gauge(comp, OCHRE), row=1, col=2)
    _apply_base(fig, 'Risk Gauges', height=280)
    return fig


# ---------------------------------------------------------------------------
# 2. trajectory_chart
# ---------------------------------------------------------------------------
def trajectory_chart(patient_data: dict) -> go.Figure:
    """5-year biomarker trajectory subplots with reference bands and trend lines."""
    biomarkers = ['ALB', 'BMI', 'HDL', 'MAP', 'eGFR']
    refs = {
        'ALB': (35, 55, 'g/L'),
        'BMI': (18.5, 24.9, 'kg/m²'),
        'HDL': (1.0, 1.6, 'mmol/L'),
        'MAP': (70, 100, 'mmHg'),
        'eGFR': (60, 120, 'mL/min/1.73m²'),
    }
    colors = [SKY, OCHRE, SAGE, ROSE, SLATE]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=biomarkers,
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]

    years = list(range(6))  # 0-5

    for idx, bm in enumerate(biomarkers):
        row, col = positions[idx]
        lo, hi, unit = refs[bm]
        vals = patient_data.get(bm, [None]*6)
        vals = [v if v is not None else np.nan for v in vals]

        # Reference band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=[hi]*6 + [lo]*6,
            fill='toself',
            fillcolor='rgba(184,200,216,0.20)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
        ), row=row, col=col)

        # Trend line (linear fit over non-nan)
        valid = [(x, v) for x, v in zip(years, vals) if not np.isnan(v)]
        if len(valid) >= 2:
            xs, ys = zip(*valid)
            m, b = np.polyfit(xs, ys, 1)
            trend = [m*x + b for x in years]
            fig.add_trace(go.Scatter(
                x=years, y=trend,
                mode='lines',
                line=dict(color=colors[idx], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ), row=row, col=col)

        # Actual values
        fig.add_trace(go.Scatter(
            x=years, y=vals,
            mode='lines+markers',
            name=bm,
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6, color=colors[idx]),
            hovertemplate=f'{bm}: %{{y:.1f}} {unit}<extra></extra>',
        ), row=row, col=col)

    # Hide unused 6th subplot
    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)

    _apply_base(fig, '5-Year Biomarker Trajectories', height=480)
    fig.update_xaxes(title_text='Year', tickvals=years)
    return fig


# ---------------------------------------------------------------------------
# 3. group_radar
# ---------------------------------------------------------------------------
def group_radar(cox_groups: dict) -> go.Figure:
    """GBTM trajectory group radar chart."""
    categories = list(next(iter(cox_groups.values())).keys()) if cox_groups else []
    if not categories:
        return go.Figure()

    group_colors = [SKY, OCHRE, ROSE, SAGE]
    fig = go.Figure()

    for i, (grp, vals) in enumerate(cox_groups.items()):
        r = [vals.get(c, 0) for c in categories]
        r.append(r[0])  # close polygon
        theta = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta,
            fill='toself',
            name=grp,
            line=dict(color=group_colors[i % len(group_colors)], width=2),
            fillcolor=group_colors[i % len(group_colors)].replace('#', 'rgba(') + ',0.12)',
        ))

    _apply_base(fig, 'GBTM Trajectory Group Profiles', height=420)
    fig.update_layout(
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(visible=True, showticklabels=True, gridcolor=MIST),
            angularaxis=dict(gridcolor=MIST),
        ),
        legend=dict(orientation='h', y=-0.12),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. counterfactual_bar
# ---------------------------------------------------------------------------
def counterfactual_bar(
    orig_mace: float, orig_comp: float,
    cf_mace: float, cf_comp: float,
    label: str = 'Intervention',
) -> go.Figure:
    """Before/after bar chart for counterfactual risk comparison."""
    categories = ['MACE Risk', 'Complication Risk']
    before = [orig_mace * 100, orig_comp * 100]
    after  = [cf_mace * 100,  cf_comp * 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before',
        x=categories, y=before,
        marker_color=ROSE,
        text=[f'{v:.1f}%' for v in before],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name=label,
        x=categories, y=after,
        marker_color=SAGE,
        text=[f'{v:.1f}%' for v in after],
        textposition='outside',
    ))

    _apply_base(fig, f'Counterfactual: {label}', height=360)
    fig.update_layout(
        barmode='group',
        yaxis=dict(title='Risk (%)', range=[0, max(before + after) * 1.25], gridcolor=MIST),
        xaxis=dict(gridcolor=_BG),
        legend=dict(orientation='h', y=1.08),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. risk_score_card
# ---------------------------------------------------------------------------
def risk_score_card(
    score: float, level: str,
    mace: float, comp: float,
    egfr_slope: float, map_slope: float,
) -> go.Figure:
    """Composite risk indicator card."""
    level_color = {
        'Low': SAGE, 'Moderate': OCHRE, 'High': ROSE, 'Very High': NAVY,
    }.get(level, SLATE)

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=round(score, 2),
        title=dict(text=f'<b>CKM Risk Score</b><br><span style="color:{level_color}">{level}</span>',
                   font=dict(size=16)),
        number=dict(font=dict(size=48, color=level_color)),
        delta=dict(reference=0.5, increasing=dict(color=ROSE), decreasing=dict(color=SAGE)),
        domain=dict(x=[0, 0.5], y=[0.4, 1.0]),
    ))

    # Mini metrics as annotations
    metrics = [
        ('MACE', f'{mace*100:.1f}%', ROSE),
        ('Comp', f'{comp*100:.1f}%', OCHRE),
        ('eGFR slope', f'{egfr_slope:+.2f}/yr', SLATE),
        ('MAP slope', f'{map_slope:+.2f}/yr', SKY),
    ]
    for i, (name, val, color) in enumerate(metrics):
        x = 0.55 + (i % 2) * 0.22
        y = 0.75 - (i // 2) * 0.35
        fig.add_annotation(
            x=x, y=y, xref='paper', yref='paper',
            text=f'<b style="color:{color}">{val}</b><br><span style="font-size:11px">{name}</span>',
            showarrow=False, align='center',
            font=dict(size=14, family=_FONT),
        )

    _apply_base(fig, '', height=260)
    return fig


# ---------------------------------------------------------------------------
# 6. risk_timeline
# ---------------------------------------------------------------------------
def risk_timeline(
    mace_risk: float, comp_risk: float,
    score: float, level: str,
) -> go.Figure:
    """Horizontal risk band timeline."""
    fig = go.Figure()

    # Background risk bands
    bands = [
        (0, 0.3,  'rgba(78,139,111,0.18)',  'Low'),
        (0.3, 0.6, 'rgba(160,120,48,0.18)', 'Moderate'),
        (0.6, 1.0, 'rgba(160,80,96,0.18)',  'High'),
    ]
    for x0, x1, color, lbl in bands:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=color, line=dict(width=0), layer='below')
        fig.add_annotation(x=(x0+x1)/2, y=0.92, text=lbl,
                           showarrow=False, font=dict(size=10, color=NAVY),
                           xref='x', yref='y')

    # Markers
    for val, name, color, ypos in [
        (mace_risk, 'MACE', ROSE, 0.55),
        (comp_risk, 'Comp', OCHRE, 0.35),
        (score,     'Score', NAVY, 0.15),
    ]:
        fig.add_trace(go.Scatter(
            x=[val], y=[ypos],
            mode='markers+text',
            marker=dict(size=16, color=color, symbol='diamond'),
            text=[f'{name}<br>{val:.2f}'],
            textposition='top center',
            textfont=dict(size=10, color=color),
            showlegend=False,
        ))

    _apply_base(fig, f'Risk Timeline — {level}', height=220)
    fig.update_xaxes(range=[0, 1], title='Risk Score', gridcolor=MIST)
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


# ---------------------------------------------------------------------------
# 7. cox_contribution_waterfall
# ---------------------------------------------------------------------------
def cox_contribution_waterfall(contributions: dict, outcome_label: str = 'MACE') -> go.Figure:
    """SHAP-style horizontal waterfall chart of Cox model feature contributions."""
    if not contributions:
        return go.Figure()

    items = sorted(contributions.items(), key=lambda x: abs(x[1]))
    features = [k for k, _ in items]
    values   = [v for _, v in items]
    colors   = [ROSE if v > 0 else SAGE for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.3f}' for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:+.3f}<extra></extra>',
    ))

    _apply_base(fig, f'Feature Contributions — {outcome_label}', height=max(300, len(features) * 28 + 80))
    fig.update_xaxes(title='Log-HR Contribution', zeroline=True, zerolinecolor=NAVY,
                     zerolinewidth=1.5, gridcolor=MIST)
    fig.update_yaxes(gridcolor=_BG)
    return fig


# ---------------------------------------------------------------------------
# 8. clpm_path_diagram
# ---------------------------------------------------------------------------
def clpm_path_diagram(clpm_df: pd.DataFrame) -> go.Figure:
    """
    Cross-lagged panel model path diagram.
    Accepts two formats:
      A) columns: from_var, from_wave, to_var, to_wave, beta, path_type
      B) columns: outcome, predictor, beta_std, sig, prefix  (clpm_key_paths_summary_v2.csv)
         outcome/predictor format: '{var}_{t1|t2|t3}'  e.g. 'map_t2', 'tyg_t1', 'egfr_t3'
    """
    VARS  = ['TyG', 'MAP', 'eGFR']
    WAVES = ['Wave 1', 'Wave 2', 'Wave 3']
    WAVE_MAP = {'t1': 'Wave 1', 't2': 'Wave 2', 't3': 'Wave 3'}
    VAR_MAP  = {'tyg': 'TyG', 'map': 'MAP', 'egfr': 'eGFR', 'fpg': 'TyG'}

    path_colors = {
        'AR':       STEEL,
        'TyG_MAP':  OCHRE,
        'MAP_eGFR': '#7B5EA7',
        'TyG_eGFR': SAGE,
    }

    # ── Format detection & normalisation ─────────────────────────────────────
    def _parse_varwave(s):
        """'map_t2' → ('MAP', 'Wave 2')"""
        parts = str(s).rsplit('_', 1)
        if len(parts) != 2:
            return None, None
        var  = VAR_MAP.get(parts[0].lower())
        wave = WAVE_MAP.get(parts[1].lower())
        return var, wave

    def _path_type(fv, tv):
        if fv == tv:                          return 'AR'
        if fv == 'TyG'  and tv == 'MAP':     return 'TyG_MAP'
        if fv == 'MAP'  and tv == 'eGFR':    return 'MAP_eGFR'
        if fv == 'TyG'  and tv == 'eGFR':    return 'TyG_eGFR'
        return 'AR'

    rows_parsed = []
    if clpm_df is not None and not clpm_df.empty:
        if 'from_var' in clpm_df.columns:
            # Format A — already structured
            for _, r in clpm_df.iterrows():
                rows_parsed.append({
                    'fv': r['from_var'], 'fw': r['from_wave'],
                    'tv': r['to_var'],   'tw': r['to_wave'],
                    'beta': float(r.get('beta', 0)),
                    'ptype': r.get('path_type', 'AR'),
                    'sig': r.get('sig', ''),
                })
        elif 'outcome' in clpm_df.columns and 'predictor' in clpm_df.columns:
            # Format B — parse from outcome/predictor strings
            # Deduplicate: keep Full Cohort rows, or first occurrence
            seen = set()
            for _, r in clpm_df.iterrows():
                fv, fw = _parse_varwave(r['predictor'])
                tv, tw = _parse_varwave(r['outcome'])
                if not all([fv, fw, tv, tw]):
                    continue
                key = (fv, fw, tv, tw)
                if key in seen:
                    continue
                seen.add(key)
                rows_parsed.append({
                    'fv': fv, 'fw': fw, 'tv': tv, 'tw': tw,
                    'beta': float(r.get('beta_std', r.get('beta', 0))),
                    'ptype': _path_type(fv, tv),
                    'sig': r.get('sig', ''),
                })

    # Node positions
    node_x = {w: i * 2.5 for i, w in enumerate(WAVES)}
    node_y = {v: (2 - i) * 2.0 for i, v in enumerate(VARS)}

    fig = go.Figure()
    annotations = []

    for row in rows_parsed:
        try:
            fv, fw = row['fv'], row['fw']
            tv, tw = row['tv'], row['tw']
            beta   = row['beta']
            ptype  = row['ptype']
            sig    = row['sig']
            color  = path_colors.get(ptype, STEEL)

            x0 = node_x[fw]; y0 = node_y[fv]
            x1 = node_x[tw]; y1 = node_y[tv]
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2

            # Offset label slightly for AR paths to avoid overlap
            offset = 0.18 if fv == tv else 0.22

            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                arrowhead=2, arrowsize=1.2,
                arrowwidth=2.5 if sig and sig != 'ns' else 1.2,
                arrowcolor=color, showarrow=True,
            ))
            annotations.append(dict(
                x=mx, y=my + offset,
                xref='x', yref='y',
                text=f'β={beta:.3f}{sig}',
                showarrow=False,
                font=dict(size=9, color=color, family=_FONT),
                bgcolor='rgba(255,255,255,0.7)',
            ))
        except Exception:
            continue

    # Draw nodes
    node_xs, node_ys, node_texts = [], [], []
    for v in VARS:
        for w in WAVES:
            node_xs.append(node_x[w])
            node_ys.append(node_y[v])
            node_texts.append(f'<b>{v}</b><br><span style="font-size:9px">{w}</span>')

    fig.add_trace(go.Scatter(
        x=node_xs, y=node_ys,
        mode='markers+text',
        marker=dict(size=38, color=MIST, line=dict(color=SLATE, width=2)),
        text=node_texts,
        textposition='middle center',
        textfont=dict(size=10, color=NAVY, family=_FONT),
        showlegend=False,
        hoverinfo='skip',
    ))

    for ptype, color in path_colors.items():
        label = {'AR': 'Autoregressive', 'TyG_MAP': 'TyG→MAP',
                 'MAP_eGFR': 'MAP→eGFR', 'TyG_eGFR': 'TyG→eGFR'}[ptype]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=color, width=3),
            name=label, showlegend=True,
        ))

    _apply_base(fig, 'Cross-Lagged Panel Model — TyG → MAP → eGFR Cascade', height=480)
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(visible=False, range=[-0.5, 5.5]),
        yaxis=dict(visible=False, range=[-0.5, 4.5]),
        legend=dict(orientation='h', y=-0.06, font=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. shap_lollipop
# ---------------------------------------------------------------------------
def shap_lollipop(shap_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    SHAP feature importance lollipop chart by trajectory group.
    Accepts columns: trajectory_group/group, feature, mean_abs_shap/shap_value
    4 groups, color-coded dots + stems, offset vertically.
    """
    if shap_df is None or shap_df.empty:
        return go.Figure()

    # Detect column names
    grp_col  = 'trajectory_group' if 'trajectory_group' in shap_df.columns else 'group'
    feat_col = 'feature'
    val_col  = 'mean_abs_shap' if 'mean_abs_shap' in shap_df.columns else 'shap_value'

    group_colors = [SKY, OCHRE, ROSE, SAGE]
    group_labels = {1: 'P1: Low-burden', 2: 'P2: Adiposity-BP', 3: 'P3: Glucose-TG', 4: 'P4: Lean-Renal'}
    groups = sorted(shap_df[grp_col].unique())[:4]

    # Pick top_n features by mean |SHAP| across groups
    mean_abs = shap_df.groupby(feat_col)[val_col].mean()
    top_features = mean_abs.nlargest(top_n).sort_values(ascending=True).index.tolist()
    df = shap_df[shap_df[feat_col].isin(top_features)].copy()

    fig = go.Figure()
    n_groups = len(groups)
    offsets = np.linspace(-0.28, 0.28, n_groups)

    for gi, (grp, color) in enumerate(zip(groups, group_colors)):
        gdf = df[df[grp_col] == grp].set_index(feat_col)

        for fi, feat in enumerate(top_features):
            val = float(gdf.loc[feat, val_col]) if feat in gdf.index else 0.0
            y_pos = fi + offsets[gi]
            fig.add_trace(go.Scatter(
                x=[0, val], y=[y_pos, y_pos],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False,
                hoverinfo='skip',
            ))

        vals = [float(gdf.loc[f, val_col]) if f in gdf.index else 0.0
                for f in top_features]
        y_positions = [i + offsets[gi] for i in range(len(top_features))]
        lbl = group_labels.get(grp, f'Group {grp}')
        fig.add_trace(go.Scatter(
            x=vals, y=y_positions,
            mode='markers',
            marker=dict(size=9, color=color, line=dict(color='white', width=1.5)),
            name=lbl,
            hovertemplate=f'{lbl}: %{{x:.3f}}<extra></extra>',
        ))

    _apply_base(fig, f'SHAP Feature Importance by Trajectory Group (Top {top_n})', height=max(400, top_n * 32 + 80))
    fig.update_xaxes(title='SHAP Value', zeroline=True, zerolinecolor=NAVY,
                     zerolinewidth=1.5, gridcolor=MIST)
    fig.update_yaxes(
        tickvals=list(range(len(top_features))),
        ticktext=top_features,
        gridcolor=_BG,
    )
    fig.update_layout(legend=dict(orientation='h', y=1.04))
    return fig


# ---------------------------------------------------------------------------
# 10. trajectory_phenotype_chart
# ---------------------------------------------------------------------------
def trajectory_phenotype_chart(traj_df: pd.DataFrame) -> go.Figure:
    """
    Multi-panel trajectory phenotype chart.
    Accepts long-format: columns traj_class, mean_val, variable, time
    OR wide-format: columns group/traj_class, time, BMI, MAP, eGFR, HDL
    4 phenotype groups, 2×2 subplot layout.
    """
    if traj_df is None or traj_df.empty:
        return go.Figure()

    # Detect long format (has 'variable' and 'mean_val' columns)
    if 'variable' in traj_df.columns and 'mean_val' in traj_df.columns:
        grp_col = 'traj_class' if 'traj_class' in traj_df.columns else 'group'
        # Pivot to wide: index=(traj_class, time), columns=variable
        df = traj_df.copy()
        df[grp_col] = df[grp_col].astype(int)
        df['variable'] = df['variable'].str.lower()
    else:
        grp_col = 'traj_class' if 'traj_class' in traj_df.columns else 'group'
        df = traj_df.copy()

    # Map variable names to display names
    var_map = {'bmi': 'BMI', 'map': 'MAP', 'egfr': 'eGFR', 'hdl': 'HDL', 'alb': 'ALB'}
    units = {'BMI': 'kg/m²', 'MAP': 'mmHg', 'eGFR': 'mL/min/1.73m²', 'HDL': 'mmol/L', 'ALB': 'g/L'}
    group_colors = [SKY, OCHRE, ROSE, SAGE]
    group_labels = {1: 'P1: Low-burden', 2: 'P2: Adiposity-BP', 3: 'P3: Glucose-TG', 4: 'P4: Lean-Renal'}

    # Get available biomarkers from data
    if 'variable' in df.columns:
        avail_vars = [v for v in ['bmi', 'map', 'egfr', 'hdl'] if v in df['variable'].unique()]
    else:
        avail_vars = [v for v in ['bmi', 'map', 'egfr', 'hdl'] if v in df.columns]

    biomarkers = [var_map.get(v, v.upper()) for v in avail_vars[:4]]
    if not biomarkers:
        biomarkers = ['BMI', 'MAP', 'eGFR', 'HDL']

    groups = sorted(df[grp_col].unique())[:4]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=biomarkers,
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for bi, (bm, raw_var) in enumerate(zip(biomarkers, avail_vars)):
        row, col = positions[bi]
        unit = units.get(bm, '')
        for gi, (grp, color) in enumerate(zip(groups, group_colors)):
            if 'variable' in df.columns:
                gdf = df[(df[grp_col] == grp) & (df['variable'] == raw_var)].sort_values('time')
                x_vals = gdf['time'].tolist()
                y_vals = gdf['mean_val'].tolist()
            else:
                gdf = df[df[grp_col] == grp].sort_values('time')
                x_vals = gdf['time'].tolist()
                y_vals = gdf[raw_var].tolist() if raw_var in gdf.columns else []

            if not y_vals:
                continue

            lbl = group_labels.get(grp, f'Group {grp}')
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                name=lbl,
                line=dict(color=color, width=2.2),
                marker=dict(size=6, color=color, line=dict(color='white', width=1.5)),
                legendgroup=f'grp{grp}',
                showlegend=(bi == 0),
                hovertemplate=f'{lbl}<br>{bm}: %{{y:.1f}} {unit}<extra></extra>',
            ), row=row, col=col)
        fig.update_yaxes(title_text=unit, row=row, col=col, gridcolor=MIST, gridwidth=0.5)
        fig.update_xaxes(title_text='Wave', row=row, col=col, gridcolor=MIST, gridwidth=0.5,
                         tickvals=[0, 1, 2], ticktext=['W1 2015', 'W2 2017', 'W3 2019'])

    _apply_base(fig, 'CKM Trajectory Phenotype Groups  (N = 95,240)', height=540)
    fig.update_layout(legend=dict(orientation='h', y=-0.10, font=dict(size=11)))
    return fig


# ---------------------------------------------------------------------------
# 11. radar_phenotype
# ---------------------------------------------------------------------------
def radar_phenotype(traj_df: pd.DataFrame) -> go.Figure:
    """
    Radar chart comparing 4 trajectory phenotype groups across 5 biomarkers.
    Accepts long-format (traj_class, mean_val, variable, time) or wide-format.
    Values are normalized 0-1 per biomarker.
    """
    if traj_df is None or traj_df.empty:
        return go.Figure()

    grp_col = 'traj_class' if 'traj_class' in traj_df.columns else 'group'
    group_labels = {1: 'P1: Low-burden', 2: 'P2: Adiposity-BP', 3: 'P3: Glucose-TG', 4: 'P4: Lean-Renal'}

    # Handle long format
    if 'variable' in traj_df.columns and 'mean_val' in traj_df.columns:
        df = traj_df.copy()
        df[grp_col] = df[grp_col].astype(int)
        df['variable'] = df['variable'].str.lower()
        avail = [v for v in ['bmi', 'map', 'egfr', 'hdl', 'alb'] if v in df['variable'].unique()]
        var_map = {'bmi': 'BMI', 'map': 'MAP', 'egfr': 'eGFR', 'hdl': 'HDL', 'alb': 'ALB'}
        biomarkers = [var_map[v] for v in avail]
        groups = sorted(df[grp_col].unique())[:4]
        # Build agg: mean per group per variable
        agg_data = {}
        for grp in groups:
            agg_data[grp] = {}
            for v, bm in zip(avail, biomarkers):
                vals = df[(df[grp_col] == grp) & (df['variable'] == v)]['mean_val']
                agg_data[grp][bm] = float(vals.mean()) if len(vals) > 0 else 0.0
        import pandas as _pd
        agg = _pd.DataFrame(agg_data).T[biomarkers]
    else:
        biomarkers = [c for c in ['BMI', 'MAP', 'eGFR', 'HDL', 'ALB'] if c in traj_df.columns]
        if not biomarkers:
            return go.Figure()
        groups = sorted(traj_df[grp_col].unique())[:4]
        agg = traj_df.groupby(grp_col)[biomarkers].mean()

    if not biomarkers:
        return go.Figure()

    group_colors = [SKY, OCHRE, ROSE, SAGE]

    # Normalize 0-1 per biomarker
    norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)

    fig = go.Figure()
    for gi, (grp, color) in enumerate(zip(groups, group_colors)):
        if grp not in norm.index:
            continue
        try:
            r = norm.loc[grp, biomarkers].tolist()
        except KeyError:
            continue
        r.append(r[0])
        theta = biomarkers + [biomarkers[0]]
        lbl = group_labels.get(grp, f'Group {grp}')
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta,
            fill='toself',
            name=lbl,
            line=dict(color=color, width=2),
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.13)",
        ))

    _apply_base(fig, 'Phenotype Radar — Normalized Biomarker Profiles', height=440)
    fig.update_layout(
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor=MIST),
            angularaxis=dict(gridcolor=MIST),
        ),
        legend=dict(orientation='h', y=-0.10),
    )
    return fig


# ---------------------------------------------------------------------------
# 12. ode_trajectory_chart
# ---------------------------------------------------------------------------
def ode_trajectory_chart(ode_df: pd.DataFrame) -> go.Figure:
    """
    ODE-simulated eGFR trajectories by intervention scenario with CI bands.
    Accepts either:
    - Long format: columns time, scenario, egfr_mean, egfr_lo, egfr_hi
    - Wide format: columns scenario, eGFR_baseline, eGFR_year10, eGFR_slope (reconstructs trajectory)
    """
    if ode_df is None or ode_df.empty:
        return go.Figure()

    scenario_styles = {
        'Protected':                 dict(color=SAGE,  dash='solid',  lw=2.5, label='Protected (early intervention)'),
        'Transitional_Untreated':    dict(color=ROSE,  dash='dot',    lw=2.0, label='Untreated natural course'),
        'Transitional_SGLT2i':       dict(color=SKY,   dash='solid',  lw=2.5, label='SGLT2i intervention'),
        'Transitional_SGLT2i_RAASi': dict(color=NAVY,  dash='solid',  lw=2.8, label='SGLT2i + RAASi combination'),
        'HighRisk_NaturalCourse':    dict(color=OCHRE, dash='dash',   lw=2.0, label='High-risk natural course'),
        # Short aliases
        'Untreated':    dict(color=ROSE,  dash='dot',   lw=2.0, label='Untreated'),
        'SGLT2i':       dict(color=SKY,   dash='solid', lw=2.5, label='SGLT2i'),
        'SGLT2i+RAASi': dict(color=NAVY,  dash='solid', lw=2.8, label='SGLT2i + RAASi'),
        'HighRisk':     dict(color=OCHRE, dash='dash',  lw=2.0, label='High-risk'),
    }

    fig = go.Figure()
    years = np.linspace(0, 10, 100)

    # Detect format
    if 'time' in ode_df.columns and 'egfr_mean' in ode_df.columns:
        # Long format
        for sc in ode_df['scenario'].unique():
            sdf = ode_df[ode_df['scenario'] == sc].sort_values('time')
            style = scenario_styles.get(sc, dict(color=STEEL, dash='solid', lw=2.0, label=sc))
            color, dash, lw = style['color'], style['dash'], style.get('lw', 2.0)
            label = style.get('label', sc)
            if 'egfr_lo' in sdf.columns and 'egfr_hi' in sdf.columns:
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig.add_trace(go.Scatter(
                    x=list(sdf['time']) + list(sdf['time'].iloc[::-1]),
                    y=list(sdf['egfr_hi']) + list(sdf['egfr_lo'].iloc[::-1]),
                    fill='toself', fillcolor=f'rgba({r},{g},{b},0.10)',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
                ))
            fig.add_trace(go.Scatter(
                x=sdf['time'], y=sdf['egfr_mean'], mode='lines',
                name=label, line=dict(color=color, width=lw, dash=dash),
                hovertemplate=f'{label}: %{{y:.1f}} mL/min<extra></extra>',
            ))
    else:
        # Wide format: reconstruct from baseline + slope
        for _, row in ode_df.iterrows():
            sc = row.get('scenario', 'Unknown')
            style = scenario_styles.get(sc, dict(color=STEEL, dash='solid', lw=2.0, label=sc))
            color, dash, lw = style['color'], style['dash'], style.get('lw', 2.0)
            label = style.get('label', sc)
            baseline = float(row.get('eGFR_baseline', row.get('egfr_baseline', 80)))
            slope    = float(row.get('eGFR_slope',    row.get('egfr_slope', -2)))
            curve    = baseline + slope * years
            ci       = np.abs(slope * 0.05) * years + 0.5
            r, g, b  = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig.add_trace(go.Scatter(
                x=list(years) + list(years[::-1]),
                y=list(curve + ci) + list((curve - ci)[::-1]),
                fill='toself', fillcolor=f'rgba({r},{g},{b},0.08)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=years, y=curve, mode='lines',
                name=label, line=dict(color=color, width=lw, dash=dash),
                hovertemplate=f'{label}: %{{y:.1f}} mL/min<extra></extra>',
            ))
            # Year-10 endpoint dot
            fig.add_trace(go.Scatter(
                x=[10], y=[float(row.get('eGFR_year10', row.get('egfr_year10', curve[-1])))],
                mode='markers', marker=dict(size=8, color=color, line=dict(color='white', width=1.5)),
                showlegend=False, hoverinfo='skip',
            ))

    # CKD threshold line
    fig.add_hline(y=60, line=dict(color=ROSE, width=1, dash='dot'),
                  annotation_text='CKD threshold (60)', annotation_position='bottom right',
                  annotation_font=dict(size=10, color=ROSE))

    _apply_base(fig, 'ODE-Simulated eGFR Trajectories by Intervention', height=420)
    fig.update_xaxes(title='Time (years)', gridcolor=MIST)
    fig.update_yaxes(title='eGFR (mL/min/1.73m²)', gridcolor=MIST)
    fig.update_layout(legend=dict(orientation='h', y=1.06))
    return fig


# ---------------------------------------------------------------------------
# 13. external_auc_chart
# ---------------------------------------------------------------------------
def external_auc_chart(ext_df: pd.DataFrame) -> go.Figure:
    """
    Forest-plot style chart for external validation.
    Supports two formats:
      - AUC format:  cohort, outcome, auc, auc_lo, auc_hi
      - RD format:   cohort, death_RD, death_CI_lo, death_CI_hi  (fig5_external_forest.csv)
    """
    if ext_df is None or ext_df.empty:
        return go.Figure()

    fig = go.Figure()

    # ── RD (risk difference) format ──────────────────────────────────────────
    if 'death_RD' in ext_df.columns:
        df = ext_df.copy().sort_values('death_RD')
        err_lo = (df['death_RD'] - df['death_CI_lo']).abs().tolist()
        err_hi = (df['death_CI_hi'] - df['death_RD']).abs().tolist()

        # colour by balance quality
        bal_map = {'YES': SAGE, 'MARGINAL': OCHRE, 'NO': ROSE}
        colors  = [bal_map.get(str(b).upper(), STEEL) for b in df.get('balance_adequate', ['YES'] * len(df))]

        fig.add_trace(go.Scatter(
            x=df['death_RD'],
            y=df['cohort'],
            mode='markers',
            marker=dict(size=10, color=colors, symbol='diamond'),
            error_x=dict(type='data', symmetric=False,
                         array=err_hi, arrayminus=err_lo,
                         thickness=1.8, width=6, color=STEEL),
            hovertemplate='<b>%{y}</b><br>RD=%{x:.4f}<extra></extra>',
            showlegend=False,
        ))

        # null line
        fig.add_vline(x=0, line=dict(color=NAVY, width=1, dash='dot'))

        # legend patches via invisible scatter
        for label, col in bal_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=9, color=col, symbol='diamond'),
                name=f'Balance: {label}',
            ))

        _apply_base(fig, 'External Validation — Risk Difference by Cohort',
                    height=max(360, len(df) * 38 + 120))
        fig.update_xaxes(title='Risk Difference (95% CI)', gridcolor=MIST, zeroline=False)
        fig.update_yaxes(gridcolor=_BG)
        fig.update_layout(legend=dict(orientation='h', y=1.06))
        return fig

    # ── AUC format (fallback) ────────────────────────────────────────────────
    outcomes = ext_df['outcome'].unique() if 'outcome' in ext_df.columns else ['AUC']
    outcome_colors = {o: c for o, c in zip(outcomes, [SKY, ROSE, OCHRE, SAGE, SLATE])}

    for outcome in outcomes:
        odf = ext_df[ext_df['outcome'] == outcome].copy() if 'outcome' in ext_df.columns else ext_df
        color = outcome_colors.get(outcome, STEEL)
        err_lo = (odf['auc'] - odf['auc_lo']).clip(lower=0).tolist() if 'auc_lo' in odf.columns else None
        err_hi = (odf['auc_hi'] - odf['auc']).clip(lower=0).tolist() if 'auc_hi' in odf.columns else None
        error_x = dict(type='data', symmetric=False,
                       array=err_hi, arrayminus=err_lo,
                       color=color, thickness=1.5, width=5) if err_lo else None
        fig.add_trace(go.Bar(
            x=odf['auc'], y=odf['cohort'], orientation='h',
            name=outcome, marker_color=color, opacity=0.85,
            error_x=error_x,
            hovertemplate='%{y} — ' + outcome + ': AUC=%{x:.3f}<extra></extra>',
        ))

    fig.add_vline(x=0.7, line=dict(color=NAVY, width=1, dash='dot'),
                  annotation_text='AUC=0.70', annotation_position='top',
                  annotation_font=dict(size=9, color=NAVY))
    _apply_base(fig, 'External Validation AUC by Cohort',
                height=max(360, len(ext_df['cohort'].unique()) * 30 + 100))
    fig.update_xaxes(title='AUC (95% CI)', range=[0.4, 1.0], gridcolor=MIST)
    fig.update_yaxes(gridcolor=_BG)
    fig.update_layout(barmode='group', legend=dict(orientation='h', y=1.06))
    return fig


# ---------------------------------------------------------------------------
# 14. bibliometrics_panel
# ---------------------------------------------------------------------------
def bibliometrics_panel(stats_dict: dict) -> dict:
    """
    Returns a dict of 3 plotly figures from a bibliometrics stats_dict.

    Expected keys:
        years           : list[int]
        counts          : list[int]
        keywords        : list[str]
        keyword_counts  : list[int]
        journals        : list[str]
        journal_counts  : list[int]

    Returns:
        {
            'trend'    : go.Figure  — publication trend bar chart,
            'keywords' : go.Figure  — keyword co-occurrence bubble chart,
            'journals' : go.Figure  — journal distribution pie chart,
        }
    """
    years          = stats_dict.get('years', [])
    counts         = stats_dict.get('counts', [])
    keywords       = stats_dict.get('keywords', [])
    keyword_counts = stats_dict.get('keyword_counts', [])
    journals       = stats_dict.get('journals', [])
    journal_counts = stats_dict.get('journal_counts', [])

    # (a) Publication trend bar chart
    fig_trend = go.Figure()
    if years and counts:
        fig_trend.add_trace(go.Bar(
            x=years, y=counts,
            marker_color=SKY,
            hovertemplate='%{x}: %{y} publications<extra></extra>',
        ))
        # Trend line
        if len(years) >= 2:
            m, b = np.polyfit(years, counts, 1)
            fig_trend.add_trace(go.Scatter(
                x=years,
                y=[m*yr + b for yr in years],
                mode='lines',
                line=dict(color=ROSE, width=2, dash='dash'),
                name='Trend',
                showlegend=True,
            ))
    _apply_base(fig_trend, 'Publication Trend', height=320)
    fig_trend.update_xaxes(title='Year', gridcolor=MIST)
    fig_trend.update_yaxes(title='Publications', gridcolor=MIST)

    # (b) Keyword co-occurrence bubble chart
    fig_kw = go.Figure()
    if keywords and keyword_counts:
        n = len(keywords)
        # Arrange in a rough grid for visual spread
        np.random.seed(42)
        x_pos = np.random.uniform(0, 10, n)
        y_pos = np.random.uniform(0, 10, n)
        max_c = max(keyword_counts) if keyword_counts else 1
        sizes = [max(8, 60 * c / max_c) for c in keyword_counts]
        colors_kw = [NAVY, SLATE, SKY, SAGE, ROSE, OCHRE, STEEL, MIST]

        fig_kw.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=[colors_kw[i % len(colors_kw)] for i in range(n)],
                opacity=0.75,
                line=dict(color='white', width=1),
            ),
            text=keywords,
            textposition='middle center',
            textfont=dict(size=9, color='white', family=_FONT),
            hovertemplate='%{text}: %{marker.size:.0f} co-occurrences<extra></extra>',
            showlegend=False,
        ))
    _apply_base(fig_kw, 'Keyword Co-occurrence', height=380)
    fig_kw.update_xaxes(visible=False)
    fig_kw.update_yaxes(visible=False)

    # (c) Journal distribution pie chart
    fig_pie = go.Figure()
    if journals and journal_counts:
        pie_colors = [NAVY, SLATE, SKY, SAGE, ROSE, OCHRE, STEEL, MIST,
                      '#2E5C8A', '#6B8FA8', '#3D7A5E', '#7A3D4C']
        fig_pie.add_trace(go.Pie(
            labels=journals,
            values=journal_counts,
            marker=dict(colors=pie_colors[:len(journals)],
                        line=dict(color='white', width=1.5)),
            textinfo='label+percent',
            textfont=dict(size=10, family=_FONT),
            hole=0.35,
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>',
        ))
    _apply_base(fig_pie, 'Journal Distribution', height=380)

    return {
        'trend':    fig_trend,
        'keywords': fig_kw,
        'journals': fig_pie,
    }
