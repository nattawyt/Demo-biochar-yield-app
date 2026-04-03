import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Biochar Yield Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtle-text {
        color: #5f6368;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .big-number {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f0fe;
        padding: 14px 16px;
        border-radius: 10px;
        color: #0b57d0;
        font-size: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6f4ea;
        padding: 14px 16px;
        border-radius: 10px;
        color: #137333;
        font-size: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .small-label {
        color: #5f6368;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .small-card-number {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌿 Biochar Yield Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Prototype version for Yield-char prediction and parameter recommendation</div>',
    unsafe_allow_html=True,
)

# =========================================================
# FIXED SEARCH GRID (hidden)
# =========================================================
T_GRID = np.arange(300, 701, 10)
RT_GRID = np.arange(10, 121, 10)
HR_GRID = np.arange(5, 31, 5)

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = "Yield_char__CatBoost.joblib"
model_path = BASE_DIR / MODEL_FILE

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

if hasattr(model, "feature_names_in_"):
    feature_order = list(model.feature_names_in_)
else:
    st.error("Cannot read feature_names_in_ from model.")
    st.stop()

# =========================================================
# SAFE DIVISION
# =========================================================
def safe_div(a, b):
    if b is None or b == 0:
        return np.nan
    return a / b

# =========================================================
# BUILD INPUT ROW
# =========================================================
def build_row(feed_dict, temp, rt, hr):
    vm = float(feed_dict["VM_bio"])
    ash = float(feed_dict["Ash_bio"])
    fc = float(feed_dict["FC_bio"])
    h = float(feed_dict["H_bio"])
    c = float(feed_dict["C_bio"])
    o = float(feed_dict["O_bio"])

    feedstock_type = "Unknown"

    hc_ratio = safe_div(h, c)
    oc_ratio = safe_div(o, c)
    vm_to_ash = safe_div(vm, ash)
    fc_to_ash = safe_div(fc, ash)

    row_map = {
        "Temp": float(temp),
        "RT": float(rt),
        "HR": float(hr),
        "VM_bio": vm,
        "Ash_bio": ash,
        "FC_bio": fc,
        "H/C_bio": hc_ratio,
        "O/C_bio": oc_ratio,
        "VM_to_Ash": vm_to_ash,
        "FC_to_Ash": fc_to_ash,
        "Temp_x_RT": float(temp) * float(rt),
        "Temp_x_HR": float(temp) * float(hr),
        "Temp_x_Ash": float(temp) * ash,
        "Temp_x_VM": float(temp) * vm,
        "Feedstock_type": feedstock_type,
        "FC_bio_derived": fc,
        "O_bio_derived": o,
    }

    missing = [col for col in feature_order if col not in row_map]
    if missing:
        raise ValueError(f"Missing features in row_map: {missing}")

    X = pd.DataFrame([[row_map[col] for col in feature_order]], columns=feature_order)
    return X

# =========================================================
# PREDICT
# =========================================================
def predict_yield(feed_dict, temp, rt, hr):
    X = build_row(feed_dict, temp, rt, hr)
    y_pred = model.predict(X)[0]
    return float(y_pred)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Feedstock properties")
st.sidebar.caption(
    "Recommended range:\n"
    "Feedstock properties = 0–100 wt.%\n\n"
    "Temp = 300–700 °C\n\n"
    "RT = 10–120 min\n\n"
    "HR = 5–30 °C/min"
)

vm_bio = st.sidebar.number_input("VM (%)", min_value=0.0, max_value=100.0, value=68.1, step=0.1)
ash_bio = st.sidebar.number_input("Ash (%)", min_value=0.0, max_value=100.0, value=4.3, step=0.1)
fc_bio = st.sidebar.number_input("FC (%)", min_value=0.0, max_value=100.0, value=10.7, step=0.1)
h_bio = st.sidebar.number_input("H (%)", min_value=0.0, max_value=100.0, value=5.6, step=0.1)
c_bio = st.sidebar.number_input("C (%)", min_value=0.0, max_value=100.0, value=57.2, step=0.1)
o_bio = st.sidebar.number_input("O (%)", min_value=0.0, max_value=100.0, value=41.4, step=0.1)

st.sidebar.markdown("## Current process settings")
temp_fixed = st.sidebar.number_input("Temperature (°C)", min_value=300, max_value=700, value=500, step=10)
rt_fixed = st.sidebar.number_input("RT (min)", min_value=10, max_value=120, value=60, step=10)
hr_fixed = st.sidebar.number_input("HR (°C/min)", min_value=5, max_value=30, value=10, step=5)

new_feed = {
    "VM_bio": vm_bio,
    "Ash_bio": ash_bio,
    "FC_bio": fc_bio,
    "H_bio": h_bio,
    "C_bio": c_bio,
    "O_bio": o_bio,
}

# =========================================================
# CURRENT PREDICTION
# =========================================================
current_yield = predict_yield(new_feed, temp_fixed, rt_fixed, hr_fixed)

st.markdown(
    '<div class="small-label">Predicted Yield-char (%) (at current settings)</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="big-number">{current_yield:.3f}</div>', unsafe_allow_html=True)

# =========================================================
# OPTIMIZE SECTION
# =========================================================
st.markdown('<div class="section-title">🎯 Optimize Key Parameters</div>', unsafe_allow_html=True)
st.write(
    "Click this button to find the best **Temp**, **RT**, and **HR** based on the current feedstock properties."
)

optimize_clicked = st.button("Suggest Best Temp, RT, and HR", use_container_width=False)

if optimize_clicked:
    with st.spinner("Searching for the best condition..."):
        results = []
        for T in T_GRID:
            for RT in RT_GRID:
                for HR in HR_GRID:
                    y = predict_yield(new_feed, T, RT, HR)
                    results.append([T, RT, HR, y])

        results_df = pd.DataFrame(results, columns=["Temp", "RT", "HR", "Yield_pred"])
        results_df = results_df.sort_values("Yield_pred", ascending=False).reset_index(drop=True)
        best = results_df.iloc[0]

        best_T = float(best["Temp"])
        best_RT = float(best["RT"])
        best_HR = float(best["HR"])
        best_y = float(best["Yield_pred"])

        temp_curve = [predict_yield(new_feed, T, best_RT, best_HR) for T in T_GRID]
        rt_curve = [predict_yield(new_feed, best_T, RT, best_HR) for RT in RT_GRID]
        hr_curve = [predict_yield(new_feed, best_T, best_RT, HR) for HR in HR_GRID]

        current_temp_y = predict_yield(new_feed, temp_fixed, best_RT, best_HR)
        current_rt_y = predict_yield(new_feed, best_T, rt_fixed, best_HR)
        current_hr_y = predict_yield(new_feed, best_T, best_RT, hr_fixed)

    st.markdown(
        f'<div class="success-box">Optimized Yield: {best_y:.3f}%</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">With your current feedstock properties, use these values to maximize yield:</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="small-label">Best Temp</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_T:.0f}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="small-label">Best RT</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_RT:.0f}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="small-label">Best HR</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_HR:.0f}</div>', unsafe_allow_html=True)

    st.markdown("### Current vs recommended")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Current setting**")
        st.write(f"Yield = {current_yield:.3f}")
        st.write(f"Temp = {temp_fixed}")
        st.write(f"RT = {rt_fixed}")
        st.write(f"HR = {hr_fixed}")
    with d2:
        st.markdown("**Recommended setting**")
        st.write(f"Yield = {best_y:.3f}")
        st.write(f"Temp = {int(best_T)}")
        st.write(f"RT = {int(best_RT)}")
        st.write(f"HR = {int(best_HR)}")

    st.markdown("### Response curves")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(T_GRID, temp_curve, marker="o")
    axes[0].axvline(best_T, linestyle="--", linewidth=1.2, label="Recommended")
    axes[0].scatter(best_T, best_y, s=60, zorder=5)
    axes[0].axvline(temp_fixed, color="red", linestyle="-", linewidth=1.0, alpha=0.85, label="Current")
    axes[0].scatter(temp_fixed, current_temp_y, color="red", s=50, zorder=6)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Predicted Yield")
    axes[0].set_title(f"Yield vs Temperature\nRT={int(best_RT)}, HR={int(best_HR)}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(RT_GRID, rt_curve, marker="o")
    axes[1].axvline(best_RT, linestyle="--", linewidth=1.2, label="Recommended")
    axes[1].scatter(best_RT, best_y, s=60, zorder=5)
    axes[1].axvline(rt_fixed, color="red", linestyle="-", linewidth=1.0, alpha=0.85, label="Current")
    axes[1].scatter(rt_fixed, current_rt_y, color="red", s=50, zorder=6)
    axes[1].set_xlabel("Residence Time (min)")
    axes[1].set_ylabel("Predicted Yield")
    axes[1].set_title(f"Yield vs Residence Time\nTemp={int(best_T)}, HR={int(best_HR)}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(HR_GRID, hr_curve, marker="o")
    axes[2].axvline(best_HR, linestyle="--", linewidth=1.2, label="Recommended")
    axes[2].scatter(best_HR, best_y, s=60, zorder=5)
    axes[2].axvline(hr_fixed, color="red", linestyle="-", linewidth=1.0, alpha=0.85, label="Current")
    axes[2].scatter(hr_fixed, current_hr_y, color="red", s=50, zorder=6)
    axes[2].set_xlabel("Heating Rate (°C/min)")
    axes[2].set_ylabel("Predicted Yield")
    axes[2].set_title(f"Yield vs Heating Rate\nTemp={int(best_T)}, RT={int(best_RT)}")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Show top 10 candidate conditions"):
        st.dataframe(results_df.head(10), use_container_width=True)