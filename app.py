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
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = "Yield_char__CatBoost.joblib"
LOGO_FILE = "logotiis.jpg"

model_path = BASE_DIR / MODEL_FILE
logo_path = BASE_DIR / LOGO_FILE

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
<style>

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}

.subtle-text {
    color: #5f6368;
    font-size: 1rem;
    margin-bottom: 0.8rem;
}

.big-number {
    font-size: 2.8rem;
    font-weight: 700;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
}

.small-label {
    color: #5f6368;
    font-size: 0.95rem;
}

.small-card-number {
    font-size: 2rem;
    font-weight: 700;
}

.success-box {
    background-color: #e6f4ea;
    padding: 18px;
    border-radius: 10px;
    color: #137333;
    font-size: 1.6rem;
    font-weight: 700;
}

.info-box {
    background-color: #e8f0fe;
    padding: 14px;
    border-radius: 10px;
    color: #0b57d0;
    font-size: 1rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HEADER WITH RIGHT LOGO
# =========================================================
col_left, col_right = st.columns([6, 1])

with col_left:

    st.markdown(
        '<div class="main-title">🌿 Biochar Yield Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtle-text">Prototype version for Yield-char prediction and parameter recommendation</div>',
        unsafe_allow_html=True
    )

with col_right:

    if logo_path.exists():
        st.image(str(logo_path), width=110)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

feature_order = list(model.feature_names_in_)

# =========================================================
# SEARCH GRID
# =========================================================
T_GRID = np.arange(300, 701, 10)
RT_GRID = np.arange(10, 181, 10)
HR_GRID = np.arange(5, 31, 5)

# =========================================================
# SAFE DIVISION
# =========================================================
def safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b

# =========================================================
# BUILD INPUT ROW
# =========================================================
def build_row(feed_dict, temp, rt, hr):

    vm = feed_dict["VM_bio"]
    ash = feed_dict["Ash_bio"]
    fc = feed_dict["FC_bio"]
    h = feed_dict["H_bio"]
    c = feed_dict["C_bio"]
    o = feed_dict["O_bio"]

    row_map = {

        "Temp": temp,
        "RT": rt,
        "HR": hr,

        "VM_bio": vm,
        "Ash_bio": ash,
        "FC_bio": fc,

        "H/C_bio": safe_div(h, c),
        "O/C_bio": safe_div(o, c),

        "VM_to_Ash": safe_div(vm, ash),
        "FC_to_Ash": safe_div(fc, ash),

        "Temp_x_RT": temp * rt,
        "Temp_x_HR": temp * hr,
        "Temp_x_Ash": temp * ash,
        "Temp_x_VM": temp * vm,

        "Feedstock_type": "Unknown",
        "FC_bio_derived": fc,
        "O_bio_derived": o
    }

    X = pd.DataFrame([[row_map[col] for col in feature_order]],
                     columns=feature_order)

    return X


# =========================================================
# PREDICT FUNCTION
# =========================================================
def predict_yield(feed_dict, temp, rt, hr):

    X = build_row(feed_dict, temp, rt, hr)

    return float(model.predict(X)[0])


# =========================================================
# SIDEBAR INPUT
# =========================================================
st.sidebar.markdown("## Feedstock properties")

vm_bio = st.sidebar.number_input("VM (%)", 0.0, 100.0, 68.1)
ash_bio = st.sidebar.number_input("Ash (%)", 0.0, 100.0, 4.3)
fc_bio = st.sidebar.number_input("FC (%)", 0.0, 100.0, 10.7)
h_bio = st.sidebar.number_input("H (%)", 0.0, 100.0, 5.6)
c_bio = st.sidebar.number_input("C (%)", 0.0, 100.0, 57.2)
o_bio = st.sidebar.number_input("O (%)", 0.0, 100.0, 41.4)

st.sidebar.markdown("## Current process settings")

temp_fixed = st.sidebar.number_input("Temperature (°C)", 300, 700, 500)
rt_fixed = st.sidebar.number_input("RT (min)", 10, 180, 60)
hr_fixed = st.sidebar.number_input("HR (°C/min)", 5, 30, 10)

optimize_clicked = st.sidebar.button("🚀 Run optimization")

# =========================================================
# CURRENT PREDICTION
# =========================================================
feed = {

    "VM_bio": vm_bio,
    "Ash_bio": ash_bio,
    "FC_bio": fc_bio,
    "H_bio": h_bio,
    "C_bio": c_bio,
    "O_bio": o_bio
}

current_yield = predict_yield(feed,
                              temp_fixed,
                              rt_fixed,
                              hr_fixed)

st.markdown(
    '<div class="small-label">Predicted Yield-char (%) (current settings)</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="big-number">{current_yield:.3f}</div>',
    unsafe_allow_html=True
)

# =========================================================
# OPTIMIZATION
# =========================================================
st.markdown(
    '<div class="section-title">🎯 Optimize Key Parameters</div>',
    unsafe_allow_html=True
)

if optimize_clicked:

    with st.spinner("Searching optimal condition..."):

        results = []

        for T in T_GRID:
            for RT in RT_GRID:
                for HR in HR_GRID:

                    y = predict_yield(feed, T, RT, HR)

                    results.append([T, RT, HR, y])

        results_df = pd.DataFrame(results,
                                  columns=["Temp", "RT", "HR", "Yield_pred"])

        results_df.sort_values("Yield_pred",
                               ascending=False,
                               inplace=True)

        best = results_df.iloc[0]

        best_T = best["Temp"]
        best_RT = best["RT"]
        best_HR = best["HR"]
        best_y = best["Yield_pred"]

    st.markdown(
        f'<div class="success-box">Optimized Yield: {best_y:.3f}%</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="small-label">Best Temp</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_T}</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="small-label">Best RT</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_RT}</div>',
                    unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="small-label">Best HR</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="small-card-number">{best_HR}</div>',
                    unsafe_allow_html=True)

    # =========================================================
    # RESPONSE CURVES
    # =========================================================
    temp_curve = [predict_yield(feed, T, best_RT, best_HR) for T in T_GRID]
    rt_curve = [predict_yield(feed, best_T, RT, best_HR) for RT in RT_GRID]
    hr_curve = [predict_yield(feed, best_T, best_RT, HR) for HR in HR_GRID]

    current_temp_y = predict_yield(feed, temp_fixed, best_RT, best_HR)
    current_rt_y = predict_yield(feed, best_T, rt_fixed, best_HR)
    current_hr_y = predict_yield(feed, best_T, best_RT, hr_fixed)

    st.markdown("### Response curves")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Temperature plot
    axes[0].plot(T_GRID, temp_curve, marker="o")
    axes[0].axvline(best_T, linestyle="--", label="Recommended")
    axes[0].scatter(best_T, best_y, s=60)
    axes[0].axvline(temp_fixed, color="red", label="Current")
    axes[0].scatter(temp_fixed, current_temp_y, color="red", s=50)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Predicted Yield")
    axes[0].set_title("Yield vs Temperature")
    axes[0].legend()
    axes[0].grid(True)

    # RT plot
    axes[1].plot(RT_GRID, rt_curve, marker="o")
    axes[1].axvline(best_RT, linestyle="--", label="Recommended")
    axes[1].scatter(best_RT, best_y, s=60)
    axes[1].axvline(rt_fixed, color="red", label="Current")
    axes[1].scatter(rt_fixed, current_rt_y, color="red", s=50)
    axes[1].set_xlabel("Residence Time (min)")
    axes[1].set_ylabel("Predicted Yield")
    axes[1].set_title("Yield vs Residence Time")
    axes[1].legend()
    axes[1].grid(True)

    # HR plot
    axes[2].plot(HR_GRID, hr_curve, marker="o")
    axes[2].axvline(best_HR, linestyle="--", label="Recommended")
    axes[2].scatter(best_HR, best_y, s=60)
    axes[2].axvline(hr_fixed, color="red", label="Current")
    axes[2].scatter(hr_fixed, current_hr_y, color="red", s=50)
    axes[2].set_xlabel("Heating Rate (°C/min)")
    axes[2].set_ylabel("Predicted Yield")
    axes[2].set_title("Yield vs Heating Rate")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Show top 10 candidate conditions"):
        st.dataframe(results_df.head(10),
                     use_container_width=True)
