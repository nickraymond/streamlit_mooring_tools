# app.py
# Deeper Moorings — Smart Mooring Cable (Bending Stress MVP)
# Run: streamlit run app.py
# Req: pip install streamlit matplotlib numpy pandas

import io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# Deeper Moorings: Theme/CSS (inline MVP)
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
    box-shadow: var(--shadow-card); border:1px solid var(--color-border);}
  .muted{color:var(--color-muted)}
  .h2{font-size:22px; font-weight:600; margin:0 0 8px 0}
  .caption{font-size:14px; color:var(--color-muted); letter-spacing:.02em}
  .smallcaps{font-variant: all-small-caps; letter-spacing: .04em; color:var(--color-muted)}
  header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer {display: none;}
</style>
"""

def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)

def topbar(project:str, depth_m:str|int, cable:str, visible:bool=True):
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

# -------------------------------
# App config + theme
# -------------------------------
st.set_page_config(page_title="Deeper Moorings — Cable Bending Stress", layout="wide")
inject_theme()
topbar("HMB", 50, "SM_cable_bottom")

# -------------------------------
# Hard-coded reference data (simple MVP)
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
MAT_DB = {
    "Cu ETP annealed (E≈110 GPa)": {"E_GPa":110.0},
    "Cu hard-drawn (E≈120 GPa)":  {"E_GPa":120.0},
    "Cu C11000 general (E≈115 GPa)": {"E_GPa":115.0},
}

def lbf_to_N(x): return x * 4.44822

def compute_curves(
    R_m, E_GPa, y_mm, n_cond, A_c_mm2,
    axial_loads, loads_unit, f_axial_share,
    helix_enable, helix_angle_deg, bend_amp_factor,
    axial_reduction_enable, axial_reduction_factor,
    safety_factor
):
    E = E_GPa * 1e9
    y = y_mm / 1000.0
    A_c = A_c_mm2 / 1e6
    kappa = 1.0 / R_m

    # Helix projection
    cos2 = 1.0
    if helix_enable:
        a = np.deg2rad(helix_angle_deg)
        cos2 = np.cos(a)**2

    sigma_b = E * kappa * y
    sigma_b = sigma_b * cos2 * bend_amp_factor

    curves = {"Case 1: Pure bending": np.abs(sigma_b) * safety_factor}

    for L in axial_loads:
        T = L if loads_unit == "N" else lbf_to_N(L)
        T_eff = T * f_axial_share
        if axial_reduction_enable:
            T_eff *= axial_reduction_factor
        sigma_ax = (T_eff / max(n_cond,1)) / max(A_c,1e-12)
        sigma_ax *= cos2
        curves[f"Case 2: + {L:g} {loads_unit} axial"] = np.abs(sigma_b + sigma_ax) * safety_factor
    return curves

# -------------------------------
# Layout (Inputs left; Plot right)
# -------------------------------
left, right = st.columns([1,2], gap="large")

with left:
    st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Inputs</div>", unsafe_allow_html=True)
    st.caption("Conductor & material")

    # Presets (hard-coded)
    awg_label = st.selectbox("AWG preset (diameter in mm shown)",
                             options=list(AWG_DB.keys()),
                             index=list(AWG_DB.keys()).index("16 AWG (1.291 mm)"))
    diam_mm = st.number_input("Conductor diameter, mm", value=float(AWG_DB[awg_label]["diam_mm"]), min_value=0.1, step=0.01)
    area_mm2 = st.number_input("Conductor metal area, mm²", value=float(AWG_DB[awg_label]["area_mm2"]), min_value=0.1, step=0.01)
    n_cond = st.number_input("Conductors sharing axial load (n)", value=2, min_value=1, step=1)

    mat_label = st.selectbox("Material", options=list(MAT_DB.keys()),
                             index=list(MAT_DB.keys()).index("Cu ETP annealed (E≈110 GPa)"))
    E_GPa = st.number_input("Young’s modulus (GPa)", value=float(MAT_DB[mat_label]["E_GPa"]), min_value=50.0, max_value=150.0, step=1.0)

    st.caption("Geometry")
    y_mm = st.number_input("y: neutral axis → conductor centroid (mm)", value=3.5, min_value=0.0, step=0.1,
                           help="Dominant driver for bending stress: ε_b = κ·y, σ_b = E·κ·y")

    st.caption("Axial loads")
    loads_unit = st.radio("Units", options=["lbf","N"], index=0, horizontal=True)
    default_loads = [20,40,60,80,100,200]
    loads_str = st.text_input(f"Axial loads (comma-separated) [{loads_unit}]",
                              value=",".join(str(x) for x in default_loads))
    try:
        axial_loads = [float(x.strip()) for x in loads_str.split(",") if x.strip()!=""]
    except Exception:
        axial_loads = default_loads

    f_axial_share = st.slider("Axial load share carried by copper (0–1)", 0.0, 1.0, 1.0, 0.05,
                              help="Set < 1.0 if strength members carry tension")

    colA, colB = st.columns(2)
    with colA:
        axial_reduction_enable = st.checkbox("Apply axial reduction factor", value=False)
    with colB:
        axial_reduction_factor = st.number_input("Reduction factor", value=0.8, min_value=0.0, step=0.05)

    st.caption("Helix / twisted pair (optional)")
    helix_enable = st.checkbox("Enable helix projection (cos²α) & bending amplification", value=False)
    helix_angle_deg = st.slider("Lay angle α (deg)", 0, 45, 30, 1)
    bend_amp_factor = st.number_input("Bending amplification k_bend", value=1.00, min_value=1.00, step=0.05,
                                      help=">1.0 if clamps/stiffeners concentrate curvature")

    st.caption("Limits & safety")
    strain_limit_pct = st.number_input("Strain limit (%)", value=0.10, min_value=0.01, step=0.01,
                                       help="~0.10% ≈ conservative elastic limit for soft copper")
    allow_tie = st.checkbox("σ_allow = E·ε_limit", value=True)
    allowable_stress_MPa = st.number_input("Allowable stress (MPa)", value=80.0, step=5.0)
    safety_factor = st.number_input("Overall safety factor (×)", value=1.00, min_value=1.00, step=0.05)

    st.caption("Radius axis")
    R_min = st.number_input("R_min (m)", value=0.2, min_value=0.05, step=0.05)
    R_max = st.number_input("R_max (m)", value=10.0, min_value=0.2, step=0.5)
    n_pts = st.slider("Samples across radius range", 100, 1000, 400, 50)

    st.markdown("</div>", unsafe_allow_html=True)  # close card

with right:
    st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Stress vs Minimum Bend Radius</div>", unsafe_allow_html=True)
    st.caption("Von Mises equals |normal stress| here (uniaxial axial + bending).")

    # Compute curves
    R = np.linspace(R_min, R_max, int(n_pts))
    curves = compute_curves(
        R_m=R, E_GPa=E_GPa, y_mm=y_mm, n_cond=n_cond, A_c_mm2=area_mm2,
        axial_loads=axial_loads, loads_unit=loads_unit, f_axial_share=f_axial_share,
        helix_enable=helix_enable, helix_angle_deg=helix_angle_deg, bend_amp_factor=bend_amp_factor,
        axial_reduction_enable=axial_reduction_enable, axial_reduction_factor=axial_reduction_factor,
        safety_factor=safety_factor
    )

    # Allowable lines
    eps_lim = strain_limit_pct / 100.0
    sigma_allow = (E_GPa * 1e9) * eps_lim if allow_tie else (allowable_stress_MPa * 1e6)

    # Plot
    fig, ax = plt.subplots(figsize=(9,6))
    labels_sorted = sorted(curves.keys(), key=lambda s: (0 if "Pure bending" in s else 1, s))
    for label in labels_sorted:
        y_curve = curves[label] / 1e6  # MPa
        ax.plot(R, y_curve, label=label)
    ax.axhline(sigma_allow/1e6, linestyle="--", linewidth=1.0, label=f"σ_allow = {sigma_allow/1e6:.1f} MPa")
    ax.axhline((E_GPa*1e9*0.001)/1e6, linestyle=":", linewidth=1.0, label="~0.1% elastic limit (E·0.001)")
    ax.set_xlabel("Minimum bend radius R (m)")
    ax.set_ylabel("Stress (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=False)

    # Quick readback table at a few radii
    radii_report = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    rows = []
    for label in labels_sorted:
        for r in radii_report:
            if R_min <= r <= R_max:
                stress_pa = np.interp(r, R, curves[label])
                rows.append({"Curve": label, "R (m)": r, "Stress (MPa)": stress_pa/1e6})
    if rows:
        st.markdown("<div class='smallcaps'>quick readback</div>", unsafe_allow_html=True)
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
    # Downloads (simple MVP)
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download plot (PNG)", data=png.getvalue(), file_name="bending_stress_plot.png", mime="image/png")

    st.markdown("</div>", unsafe_allow_html=True)  # close card

# -------------------------------
# Quick Summary & Equations (same page, below)
# -------------------------------
st.markdown("<div class='dmui-card'>", unsafe_allow_html=True)
st.markdown("<div class='h2'>Quick Summary & Equations</div>", unsafe_allow_html=True)
st.markdown("""
**Geometry & Curvature**  
- Curvature–radius:  \\\( \\kappa = 1/R \\\) (1/m)  
- Conductor offset from neutral axis:  \\\( y \\\) (m)

**Bending strain & stress (linear elastic)**  
- \\\( \\varepsilon_b = \\kappa\\,y \\\)  
- \\\( \\sigma_b = E\\,\\varepsilon_b = E\\,\\kappa\\,y \\\)

**Axial + bending (superposition; uniaxial)**  
- \\\( \\sigma_{ax} = \\dfrac{T\\,f_{\\text{share}}}{n\\,A_c} \\\)  
- \\\( \\sigma_{\\text{tot}} = \\sigma_{ax} + \\sigma_b \\\)  
- Von Mises here: \\\( \\sigma_{\\text{VM}} = |\\sigma_{\\text{tot}}| \\\)

**Helix / twisted pair (optional)**  
- Projection to strand axis: \\\( \\cos^2\\alpha \\\)  
- Local curvature amplification (clamps/stiffeners): \\\( k_{\\text{bend}} \\ge 1 \\\)  
- If enabled: \\\( \\sigma_b \\leftarrow \\sigma_b\\,\\cos^2\\alpha\\,k_{\\text{bend}},\\quad \\sigma_{ax} \\leftarrow \\sigma_{ax}\\,\\cos^2\\alpha \\\)

**Design guardrails**  
- Strain limit (default): \\(\\varepsilon_{\\text{limit}}\\approx0.1\\%\\Rightarrow E\\,\\varepsilon_{\\text{limit}}\\)  
- Allowable stress line (set from strain or direct MPa)
- Sensitivity: \\(\\sigma_b\\) is **linear** in \\(y\\) and **inverse** in \\(R\\). Keep **\\(y\\)** small and **\\(R\\)** large.
""")
st.markdown("</div>", unsafe_allow_html=True)
