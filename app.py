# app.py
# Deeper Moorings — Smart Mooring Cable (Cross-Section + Bending Stress)
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import streamlit as st

# Plotly (interactive chart)
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# -------------------------------
# Theme / CSS
# -------------------------------
CSS = """
<style>
  :root{
    --color-primary:#2563EB; --color-primary-200:#60A5FA;
    --color-text:#111827; --color-muted:#6B7280;
    --color-bg:#F3F4F6; --color-card:#FFFFFF; --color-border:#E5E7EB;
    --radius-sm:8px; --radius:12px; --radius-lg:16px;
    --shadow-soft:0 6px 16px rgba(0,0,0,.06);
    --shadow-card:0 10px 20px rgba(0,0,0,.05);
  }
  @import url('https://rsms.me/inter/inter.css');
  html {font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial;}
  body {color: var(--color-text);}
  .block-container{padding-top: 12px;}
  .dmui-topbar{
    background:linear-gradient(90deg,var(--color-primary),var(--color-primary-200));
    color:#fff; padding:10px 14px; border-radius: var(--radius);
    box-shadow: var(--shadow-card); margin-bottom:12px;
  }
  .dmui-topbar .row{display:flex; justify-content:space-between; align-items:center; gap:12px}
  .dmui-pill{background:rgba(255,255,255,.22); padding:2px 8px; border-radius:999px; font-size:.85rem}
  .dmui-card{background:var(--color-card); border-radius: var(--radius-lg); padding:16px;
    box-shadow: var(--shadow-card); border:1px solid var(--color-border); margin-bottom:12px;}
  .muted{color:var(--color-muted)}
  .h2{font-size:22px; font-weight:600; margin:0 0 8px 0}
  .smallcaps{font-variant: all-small-caps; letter-spacing: .04em; color:var(--color-muted)}
  header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer {display: none;}
  .katex-display { text-align: left !important; margin: 0.15rem 0 !important; }
  .katex-display > .katex { text-align: left !important; }
</style>
"""
def inject_theme(): st.markdown(CSS, unsafe_allow_html=True)
def topbar(project:str, depth_m:int|str, cable:str, visible:bool=True):
    if not visible: return
    st.markdown(f"""
    <div class='dmui-topbar'>
      <div class='row'>
        <div>
          <b>Deeper Moorings</b>
          &nbsp;&nbsp;<span class='dmui-pill'>Project: {project}</span>
          &nbsp;<span class='dmui-pill'>Depth: {depth_m} m</span>
          &nbsp;<span class='dmui-pill'>Cable: {cable}</span>
        </div>
        <div class='muted'>Bending Stress MVP</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Deeper Moorings — Cable Bending Stress", layout="wide")
inject_theme(); topbar("HMB", 50, "SM_cable_bottom")

# -------------------------------
# Data
# -------------------------------
AWG_DB = {
    "12 AWG (2.053 mm)": {"diam_mm":2.053, "area_mm2":3.31},
    "14 AWG (1.628 mm)": {"diam_mm":1.628, "area_mm2":2.08},
    "16 AWG (1.291 mm)": {"diam_mm":1.291, "area_mm2":1.31},
    "10 AWG (2.588 mm)": {"diam_mm":2.588, "area_mm2":5.26},
    "8 AWG (3.264 mm)":  {"diam_mm":3.264, "area_mm2":8.37},
    "6 AWG (4.115 mm)":  {"diam_mm":4.115, "area_mm2":13.30},
    "4 AWG (5.189 mm)":  {"diam_mm":5.189, "area_mm2":21.20},
    "2 AWG (6.544 mm)":  {"diam_mm":6.544, "area_mm2":33.60},
    "1/0 AWG (8.251 mm)": {"diam_mm":8.251, "area_mm2":53.50},
    "4/0 AWG (11.684 mm)": {"diam_mm":11.684, "area_mm2":107.20},
}
# Default material: tinned + annealed OFC C10200 OSO50
MAT_DB = {
    "Cu C10200 OSO50 (annealed, tinned) — E≈115 GPa": {"E_GPa":115.0},
    "Cu ETP annealed — E≈110 GPa": {"E_GPa":110.0},
    "Cu hard-drawn — E≈120 GPa":  {"E_GPa":120.0},
}
def lbf_to_N(x): return x * 4.44822

# -------------------------------
# Mechanics
# -------------------------------
def compute_curves(R_m, E_GPa, y_mm, A_c_mm2,
                   axial_loads, loads_unit, f_axial_share,
                   helix_enable, helix_angle_deg, bend_amp_factor,
                   axial_reduction_enable, axial_reduction_factor,
                   safety_factor, n_cond:int=2):
    """Uniaxial axial + bending (VM = |sigma_tot| here)."""
    E = E_GPa * 1e9
    y = y_mm / 1000.0
    A_c = A_c_mm2 / 1e6
    kappa = 1.0 / R_m
    cos2 = 1.0
    if helix_enable:
        a = np.deg2rad(helix_angle_deg); cos2 = np.cos(a)**2
    sigma_b = E * kappa * y * cos2 * bend_amp_factor
    curves = {"Case 1: Pure bending": np.abs(sigma_b) * safety_factor}
    for L in axial_loads:
        T = L if loads_unit == "N" else lbf_to_N(L)
        T_eff = T * f_axial_share
        if axial_reduction_enable: T_eff *= axial_reduction_factor
        sigma_ax = (T_eff / n_cond) / max(A_c,1e-12) * cos2
        curves[f"Case 2: + {L:g} {loads_unit} axial"] = np.abs(sigma_b + sigma_ax) * safety_factor
    return curves

# Helper: compute default y (minimum allowed by geometry & clearance)
def compute_y_default(r_ins_m: float, r_core_m: float, clr_m: float) -> float:
    """Minimum y so the two insulated conductors (mirrored at ±y) just don't overlap,
       and still fit inside the core with clearance."""
    y_min_overlap = r_ins_m + 0.5*clr_m
    y_max_fit     = max(0.0, r_core_m - r_ins_m - clr_m)
    if y_min_overlap > y_max_fit:
        # No feasible placement; return y_max_fit and the caller will show errors
        return y_max_fit
    return y_min_overlap

# ==============================================================
# CARD 1 — Cable Geometry & Cross-Section (locked ±17.5 mm, OD≤35 mm)
# ==============================================================
st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
st.markdown("<div class='h2'>Cable Geometry & Cross-Section</div>", unsafe_allow_html=True)
cg_left, cg_right = st.columns([1,1], gap="large")

with cg_left:
    st.caption("Conductor & material")
    awg_label = st.selectbox("AWG (diameter shown)", list(AWG_DB.keys()),
                             index=list(AWG_DB.keys()).index("16 AWG (1.291 mm)"))
    diam_mm = AWG_DB[awg_label]["diam_mm"]      # from DB (no manual override)
    area_mm2 = AWG_DB[awg_label]["area_mm2"]    # from DB
    mat_label = st.selectbox("Material", list(MAT_DB.keys()),
                             index=list(MAT_DB.keys()).index("Cu C10200 OSO50 (annealed, tinned) — E≈115 GPa"))
    E_GPa = float(MAT_DB[mat_label]["E_GPa"])

    st.caption("Layout & clearances")
    ins_thk_mm = st.number_input("Conductor insulation thickness (radial), mm", value=0.75, min_value=0.0, step=0.05)
    od_mm      = st.number_input("Outer jacket OD, mm (≤ 35 mm)", value=14.5, min_value=2.0, max_value=35.0, step=0.5)
    jkt_wall_mm= st.number_input("Jacket wall thickness, mm (radial)", value=3.0, min_value=0.2, step=0.1,
                                 help="Core radius = OD/2 − wall")
    min_clear_mm = st.number_input("Min clearance (to jacket & other conductor), mm", value=0.2, min_value=0.0, step=0.1)

with cg_right:
    # Geometry (meters)
    r_outer = (od_mm/2.0) / 1000.0
    r_core  = (od_mm/2.0 - jkt_wall_mm) / 1000.0      # inner radius (filler/core)
    r_cu    = (diam_mm/2.0) / 1000.0                  # copper radius
    r_ins   = r_cu + (ins_thk_mm / 1000.0)            # insulated radius
    clr_m   = min_clear_mm / 1000.0

    # Default y = minimum allowed (avoid overlap + fit inside core)
    y_default_mm = 1000.0 * compute_y_default(r_ins, r_core, clr_m)
    # User can override (defaults to the computed minimum)
    y_mm = st.number_input(
        "y (NA → left conductor centroid), mm",
        value=float(np.round(y_default_mm, 3)), min_value=0.0, step=0.05,
        help=f"Auto-default = min allowed: max(overlap limit, fit limit). Current default ≈ {y_default_mm:.3f} mm"
    )

    # Mirrored conductors at ±y
    x_left, x_right = -y_mm/1000.0, +y_mm/1000.0

    # ---- Validations (use r_ins for fit) ----
    fit_errors = []
    if jkt_wall_mm*2.0 >= od_mm:
        fit_errors.append("Jacket wall is too large relative to OD (no room for core).")
    if r_core <= 0:
        fit_errors.append("Computed core radius ≤ 0. Increase OD or reduce wall thickness.")
    if abs(x_left) + r_ins + clr_m > r_core:
        fit_errors.append("Left insulated conductor violates core/jacket clearance. Reduce y/diameter, or increase OD/wall.")
    if abs(x_right) + r_ins + clr_m > r_core:
        fit_errors.append("Right insulated conductor violates core/jacket clearance.")
    if (2*(y_mm/1000.0)) < (2*r_ins + clr_m):
        fit_errors.append("Insulated conductors overlap (or violate min clearance). Increase y or reduce diameter/insulation.")

    # ----------------- Draw cross-section (locked to ±17.5 mm) -----------------
    fig_cs, ax_cs = plt.subplots(figsize=(5.8,5.8))
    ax_cs.set_aspect('equal'); ax_cs.axis('off')

    # Colors (Pantone 3955C-ish for jacket)
    jacket_color = "#F2F200"  # bright yellow
    filler_color = "#D1D5DB"  # light gray
    left_ins_color  = "#111827"  # dark insulation
    right_ins_color = "#9CA3AF"  # light insulation
    copper_color = "#b87333"     # copper

    # Outer jacket + core
    circ_outer = plt.Circle((0,0), r_outer, color=jacket_color, ec=jacket_color, lw=1.5)
    ax_cs.add_patch(circ_outer)
    circ_core  = plt.Circle((0,0), r_core, color=filler_color, ec="#9CA3AF", lw=1.0)
    ax_cs.add_patch(circ_core)

    # Solid insulation annulus + copper
    # Left @ -y
    left_ann = Wedge((x_left, 0), r_ins, 0, 360, width=(r_ins - r_cu),
                     facecolor=left_ins_color, edgecolor="#111827", linewidth=1.5)
    left_cu  = plt.Circle((x_left,0), r_cu, color=copper_color, ec="#111827", lw=0.8)
    ax_cs.add_patch(left_ann); ax_cs.add_patch(left_cu)
    # Right @ +y
    right_ann = Wedge((x_right, 0), r_ins, 0, 360, width=(r_ins - r_cu),
                      facecolor=right_ins_color, edgecolor="#9CA3AF", linewidth=1.5)
    right_cu  = plt.Circle((x_right,0), r_cu, color=copper_color, ec="#9CA3AF", lw=0.8)
    ax_cs.add_patch(right_ann); ax_cs.add_patch(right_cu)

    # Dashed lines: NA (x=0) and left centroid (x=-y)
    view_half = 17.5/1000.0  # ±17.5 mm window (max OD 35 mm)
    ax_cs.plot([0,0], [-view_half, view_half], ls="--", lw=1.2, color="#111827", alpha=0.7)
    ax_cs.plot([x_left,x_left], [-view_half, view_half], ls="--", lw=1.2, color="#111827", alpha=0.7)

    # Double-arrow for y at the top
    top_y = view_half * 0.92
    ax_cs.annotate("", xy=(x_left, top_y), xytext=(0, top_y),
                   arrowprops=dict(arrowstyle="<->", lw=1.2, color="#111827"))
    ax_cs.text((x_left)/2.0, top_y + 0.001, f"y = {y_mm:.2f} mm", ha="center", fontsize=10)

    # Scale bar: 5 mm
    bar_len_m = 5.0/1000.0
    x0, y0 = -view_half*0.88, -view_half*0.92
    ax_cs.plot([x0, x0+bar_len_m], [y0, y0], color="#111827", lw=2.0)
    ax_cs.text(x0 + bar_len_m/2, y0-0.001, "5 mm", ha="center", va="top", fontsize=9)

    ax_cs.set_xlim(-view_half, view_half); ax_cs.set_ylim(-view_half, view_half)
    st.pyplot(fig_cs, clear_figure=False)

    if fit_errors:
        for e in fit_errors: st.error(e)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# CARD 2 — Stress vs Minimum Bend Radius (inputs LEFT, plot RIGHT)
# ==============================================================
st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
st.markdown("<div class='h2'>Stress vs Minimum Bend Radius</div>", unsafe_allow_html=True)
s_left, s_right = st.columns([1.1,1], gap="large")

with s_left:
    st.caption("Loads & factors")
    loads_unit = st.radio("Axial load units", ["lbf","N"], index=0, horizontal=True)
    default_loads = [20,40,60,80,100,200]
    loads_str = st.text_input(f"Axial loads (comma-separated) [{loads_unit}]",
                              value=",".join(str(x) for x in default_loads))
    try:
        axial_loads = [float(x.strip()) for x in loads_str.split(",") if x.strip()!=""]
    except Exception:
        axial_loads = default_loads

    f_axial_share = st.slider("Axial load share carried by copper (0–1)", 0.0, 1.0, 1.0, 0.05)

    st.caption("Helix / amplification (optional)")
    helix_enable = st.checkbox("Enable cos²α & k_bend", value=False)
    helix_angle_deg = st.slider("Lay angle α (deg)", 0, 45, 30, 1)
    bend_amp_factor = st.number_input("Bending amplification k_bend", value=1.00, min_value=1.00, step=0.05)

    st.caption("Limits & axis")
    strain_limit_pct = st.number_input("Strain limit (%)", value=0.10, min_value=0.01, step=0.01)
    allow_tie = st.checkbox("σ_allow = E·ε_limit", value=True)
    allowable_stress_MPa = st.number_input("Allowable stress (MPa)", value=80.0, step=5.0)
    R_min = st.number_input("R_min (m)", value=0.2, min_value=0.05, step=0.05)
    R_max = st.number_input("R_max (m)", value=10.0, min_value=0.2, step=0.5)
    n_pts = st.slider("Samples across radius range", 100, 1000, 400, 50)

    st.caption("Chart options")
    hover_on = st.checkbox("Enable hover tooltips", value=True)

with s_right:
    # Build curves (assume n_cond=2)
    R = np.linspace(R_min, R_max, int(n_pts))
    curves = compute_curves(
        R_m=R, E_GPa=E_GPa, y_mm=y_mm, A_c_mm2=area_mm2,
        axial_loads=axial_loads, loads_unit=loads_unit, f_axial_share=f_axial_share,
        helix_enable=helix_enable, helix_angle_deg=helix_angle_deg, bend_amp_factor=bend_amp_factor,
        axial_reduction_enable=False, axial_reduction_factor=1.0,
        safety_factor=1.0, n_cond=2
    )
    eps_lim = strain_limit_pct / 100.0
    sigma_allow = (E_GPa * 1e9) * eps_lim if allow_tie else (allowable_stress_MPa * 1e6)

    if HAS_PLOTLY:
        fig = go.Figure()
        labels_sorted = sorted(curves.keys(), key=lambda s: (0 if "Pure bending" in s else 1, s))
        for label in labels_sorted:
            fig.add_trace(go.Scatter(
                x=R, y=curves[label]/1e6,  # MPa
                mode="lines", name=label,
                hovertemplate=None if not hover_on else "R = %{x:.3f} m<br>σ = %{y:.2f} MPa<extra>"+label+"</extra>",
                hoverinfo="skip" if not hover_on else "all",
            ))

        # Allowable lines
        fig.add_hline(y=sigma_allow/1e6, line_dash="dash", line_width=1,
                      annotation_text=f"σ_allow = {sigma_allow/1e6:.1f} MPa",
                      annotation_position="top left")
        fig.add_hline(y=(E_GPa*1e9*0.001)/1e6, line_dash="dot", line_width=1,
                      annotation_text="~0.1% elastic", annotation_position="bottom left")

        fig.update_layout(
            template="plotly_white",
            height=580,  # similar vertical size to cross-section
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            xaxis_title="Minimum bend radius R (m)",
            yaxis_title="Stress (MPa)",
            hovermode=("x unified" if hover_on else False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not found — showing a static matplotlib plot. Install with: pip install plotly")
        fig, ax = plt.subplots(figsize=(6.2,5.8))
        labels_sorted = sorted(curves.keys(), key=lambda s: (0 if "Pure bending" in s else 1, s))
        for label in labels_sorted:
            ax.plot(R, curves[label]/1e6, label=label)
        ax.axhline(sigma_allow/1e6, linestyle="--", linewidth=1.0, label=f"σ_allow = {sigma_allow/1e6:.1f} MPa")
        ax.axhline((E_GPa*1e9*0.001)/1e6, linestyle=":", linewidth=1.0, label="~0.1% elastic")
        ax.set_xlabel("Minimum bend radius R (m)"); ax.set_ylabel("Stress (MPa)")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper center", ncol=2, fontsize=9)
        st.pyplot(fig, clear_figure=False)

    # Quick readback at common radii
    radii_report = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    rows = []
    for label, arr in curves.items():
        for r in radii_report:
            if R_min <= r <= R_max:
                stress_pa = np.interp(r, R, arr)
                rows.append({"Curve": label, "R (m)": r, "Stress (MPa)": stress_pa/1e6})
    if rows:
        st.markdown("<div class='smallcaps'>quick readback</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# CARD 3 — Quick Summary & Equations (unchanged helper)
# ==============================================================
st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
st.markdown("<div class='h2'>Quick Summary & Equations</div>", unsafe_allow_html=True)
def eq_row(label: str, tex: str):
    c1, c2 = st.columns([1.2, 2.8])
    with c1: st.markdown(f"<div class='eq-row label'>{label}</div>", unsafe_allow_html=True)
    with c2: st.latex(tex)
st.markdown("<style>.eq-row .label{font-weight:600;color:var(--color-text)}.eq-row{margin:2px 0 8px 0}</style>", unsafe_allow_html=True)
eq_row("Curvature–radius:",   r"\kappa = \frac{1}{R} \quad \text{(1/m)}")
eq_row("Offset from NA:",     r"y \quad \text{(m)}")
st.markdown("**Bending strain & stress (linear elastic)**"); eq_row("Strain:", r"\varepsilon_b = \kappa\,y"); eq_row("Stress:", r"\sigma_b = E\,\varepsilon_b = E\,\kappa\,y")
st.markdown("**Axial + bending (superposition; uniaxial)**"); eq_row("Axial stress:", r"\sigma_{ax}=\frac{T\,f_{\text{share}}}{n\,A_c}"); eq_row("Total stress:", r"\sigma_{\text{tot}}=\sigma_{ax}+\sigma_b"); eq_row("Von Mises here:", r"\sigma_{\text{VM}}=\left|\sigma_{\text{tot}}\right|")
st.markdown("**Helix / twisted pair (optional)**"); eq_row("Projection:", r"\cos^2\alpha"); eq_row("Amplification:", r"k_{\text{bend}} \ge 1"); eq_row("Applied as:", r"\sigma_b \leftarrow \sigma_b\,\cos^2\alpha\,k_{\text{bend}}"); eq_row("", r"\sigma_{ax} \leftarrow \sigma_{ax}\,\cos^2\alpha")
st.markdown("**Design guardrails**"); eq_row("Ref. stress @ 0.1%:", r"\varepsilon_{\text{limit}}\approx 0.1\% \;\Rightarrow\; \sigma_{\text{ref}}=E\,\varepsilon_{\text{limit}}")
st.markdown("Sensitivity: bending stress is linear in **y** and inverse in **R**. Keep **y** small and **R** large.")
st.markdown("</div>", unsafe_allow_html=True)
