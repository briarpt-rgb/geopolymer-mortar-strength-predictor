import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    page_title="Geopolymer Mortar Strength Predictor",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
.main-header {
    background: linear-gradient(90deg, #6a00f4, #00b4d8, #00d4a6);
    padding: 26px 30px; border-radius: 18px; color: white;
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
}
.main-header h1 { margin: 0; font-size: 42px; font-weight: 800; }
.main-header p { margin: 6px 0 0 0; font-size: 18px; }
.card {
    background: #f8f9ff; border: 1px solid #d9def2; border-radius: 16px;
    padding: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 18px;
}
.result-card {
    background: linear-gradient(135deg, #0ead69, #1b9aaa);
    color: white; border-radius: 18px; padding: 18px;
    text-align: center; box-shadow: 0 8px 22px rgba(0,0,0,0.18); margin-top: 12px;
}
.result-card h1 { margin: 8px 0 0 0; font-size: 36px; font-weight: 900; }
.check-good { padding: 10px; background: #e8fff1; border-left: 5px solid #18a957; border-radius: 8px; margin-bottom: 10px; }
.check-warn { padding: 10px; background: #fff6e5; border-left: 5px solid #d98e04; border-radius: 8px; margin-bottom: 10px; }
.check-bad  { padding: 10px; background: #fff0f0; border-left: 5px solid #d62828; border-radius: 8px; margin-bottom: 10px; }
.gauge-high { padding: 12px; background: #e8fff1; border-radius: 10px; font-weight: 700; text-align: center; color: #127a43; }
.gauge-low  { padding: 12px; background: #fff0f0; border-radius: 10px; font-weight: 700; text-align: center; color: #b42318; }
div.stButton > button:first-child {
    background: linear-gradient(135deg,#ff7b00,#ff006e);
    color: white; font-weight: 700; font-size: 18px;
    border-radius: 12px; height: 55px; border: none;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
# This is the file produced by the training notebook:
#   joblib.dump(final_model, f"model_winner_{safe_name}.pkl")
# which for the winning Extra Trees model resolves to "model_winner_Extra_Trees.pkl".
MODEL_PATH = "model_winner_Extra_Trees.pkl"
model = joblib.load(MODEL_PATH)

# Tuned Extra Trees — 10-fold CV and Monte Carlo metrics
metrics_10f = {"R2": 0.906, "RMSE": 5.43,  "MAPE (%)": 10.04, "r": 0.950, "IoA": 0.973}
metrics_mc  = {"R2": 0.844, "RMSE": 6.24,  "MAPE (%)": 12.27, "r": 0.927, "IoA": 0.953}

dataset_ranges = {
    "Molarity (M)":            (3.0,   16.0),
    "Fine Aggregate (kg/m³)":  (432.0, 1500.0),
    "Nano Silica (kg/m³)":     (0.0,   50.0),
    "NaOH (kg/m³)":            (64.28, 130.0),
    "Na₂SiO₃ (kg/m³)":        (127.27, 250.0),
    "Na₂SiO₃ / NaOH":         (1.5,   2.5),
    "Alk : Binder":            (0.30,  0.70),
    "Curing Condition (°C)":   (20.0,  70.0),
    "Fly Ash (kg/m³)":         (0.0,   734.0),
    "Slag (kg/m³)":            (0.0,   650.0),
    "Metakaolin (kg/m³)":      (0.0,   450.0),
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def make_pdf(prediction, inputs, ratio, binder_total):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 800, "Geopolymer Mortar Strength Prediction Report")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 775, "Model: Extra Trees Regressor (Bayesian-tuned)")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 755, f"28-Day Compressive Strength: {prediction:.2f} MPa")
    pdf.drawString(50, 735, f"Na2SiO3/NaOH Ratio: {ratio:.3f}")
    pdf.drawString(50, 715, f"Total Binder: {binder_total:.2f} kg/m3")
    pdf.drawString(50, 695, f"Model 10-Fold R2: {metrics_10f['R2']}  |  Monte Carlo R2: {metrics_mc['R2']}")
    y = 660
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Input Parameters:")
    y -= 18
    pdf.setFont("Helvetica", 11)
    for k, v in inputs.items():
        pdf.drawString(60, y, f"{k}: {v}")
        y -= 15
    pdf.save()
    buffer.seek(0)
    return buffer


def show_check(msg, level):
    st.markdown(f'<div class="check-{level}">{msg}</div>', unsafe_allow_html=True)


# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🧪 Geopolymer Mortar Strength Predictor</h1>
    <p>Interactive machine-learning tool based on a Bayesian-tuned Extra Trees model &nbsp;|&nbsp; 28-day compressive strength</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Dataset Input Ranges")
    st.caption("Stay within these ranges for reliable predictions.")
    for k, (min_v, max_v) in dataset_ranges.items():
        st.markdown(f"**{k}**: {min_v} – {max_v}")
    st.divider()
    st.subheader("🤖 Model Info")
    st.markdown("""
    **Algorithm:** Extra Trees Regressor  
    **Optimisation:** Bayesian (Optuna TPE, 200 trials)  
    **Training set:** 75 geopolymer mortar mixes  
    **Validation:** 10-fold CV + 100-rep Monte Carlo  
    """)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([2.2, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚗️ Mixture Design Inputs")

    molarity    = st.slider("Molarity (M)",              3.0,   16.0,  10.0,  step=0.5)
    fine_agg    = st.slider("Fine Aggregate (kg/m³)",  432.0, 1500.0, 700.0,  step=10.0)
    nano_silica = st.slider("Nano Silica (kg/m³)",       0.0,   50.0,   5.0,  step=0.5)
    naoh        = st.slider("NaOH (kg/m³)",             64.28, 130.0,  80.0,  step=1.0)
    na2sio3     = st.slider("Na₂SiO₃ (kg/m³)",        127.27, 250.0, 160.0,  step=1.0)
    alk_binder  = st.slider("Alk : Binder",              0.30,   0.70,  0.45, step=0.01)
    curing      = st.slider("Curing Condition (°C)",    20.0,   70.0,  28.0,  step=1.0)

    st.markdown("#### Binder Composition")
    b1, b2, b3 = st.columns(3)
    fly_ash     = b1.number_input("Fly Ash (kg/m³)",    0.0,  734.0, 200.0)
    slag        = b2.number_input("Slag (kg/m³)",        0.0,  650.0, 100.0)
    metakaoline = b3.number_input("Metakaolin (kg/m³)",  0.0,  450.0, 250.0)

    predict = st.button("Calculate Compressive Strength", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Model performance card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")
    st.caption("Tuned Extra Trees Regressor")
    c1, c2 = st.columns(2)
    c1.metric("10-Fold R²",  f"{metrics_10f['R2']:.3f}")
    c2.metric("MC R²",       f"{metrics_mc['R2']:.3f}")
    c1.metric("10-Fold RMSE", f"{metrics_10f['RMSE']:.2f}")
    c2.metric("MC RMSE",      f"{metrics_mc['RMSE']:.2f}")
    c1.metric("10-Fold MAPE", f"{metrics_10f['MAPE (%)']:.2f}%")
    c2.metric("MC MAPE",      f"{metrics_mc['MAPE (%)']:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Live mix checks
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔬 Live Mixture Checks")
    ratio        = na2sio3 / naoh if naoh != 0 else 0
    binder_total = fly_ash + slag + metakaoline

    st.metric("Na₂SiO₃ / NaOH",  f"{ratio:.3f}")
    st.metric("Total Binder",     f"{binder_total:.2f} kg/m³")
    st.metric("Nano Silica / Binder", f"{(nano_silica / binder_total * 100):.1f} %" if binder_total > 0 else "—")

    if 1.5 <= ratio <= 2.5:
        show_check("✅ SS/SH ratio within training range", "good")
    else:
        show_check("⚠️ SS/SH ratio outside training range", "warn")

    if binder_total < 432 or binder_total > 1100:
        show_check("⚠️ Total binder outside training range", "warn")
    else:
        show_check("✅ Total binder within training range", "good")

    if nano_silica > 0 and binder_total > 0 and (nano_silica / binder_total) > 0.10:
        show_check("⚠️ Nano-silica content unusually high (>10% of binder)", "warn")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if predict:
    # Feature names MUST match exactly what the model was trained on.
    # The notebook's load_data() produces these underscored column names,
    # which sklearn stores in model.feature_names_in_:
    #   Molarity_M, Fly_ash, Slag, Metakaoline, Fine_Aggregate,
    #   Nano_Silica_Kg, NaOH, Na2SIO3, Na2SiO3_NaOH, Alk_Binder, Curing_Condition
    input_data = pd.DataFrame([{
        "Molarity_M":         molarity,
        "Fly_ash":            fly_ash,
        "Slag":               slag,
        "Metakaoline":        metakaoline,
        "Fine_Aggregate":     fine_agg,
        "Nano_Silica_Kg":     nano_silica,
        "NaOH":               naoh,
        "Na2SIO3":            na2sio3,
        "Na2SiO3_NaOH":       ratio,
        "Alk_Binder":         alk_binder,
        "Curing_Condition":   curing,
    }])

    # Reorder to match the model's expected feature order (defensive — works
    # even if a future version of load_data() reorders columns).
    if hasattr(model, "feature_names_in_"):
        input_data = input_data[list(model.feature_names_in_)]

    prediction = model.predict(input_data)[0]

    st.markdown(
        f'<div class="result-card">'
        f'<h2>Predicted 28-Day Compressive Strength</h2>'
        f'<h1>{prediction:.2f} MPa</h1>'
        f'<p style="margin:6px 0 0 0;font-size:15px;opacity:0.85;">Extra Trees Regressor &nbsp;|&nbsp; 10-fold R² = {metrics_10f["R2"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # Reliability band
    if 10 <= prediction <= 120:
        st.markdown('<div class="gauge-high">🟢 Reliability: High — prediction within training data range</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="gauge-low">🔴 Reliability: Low — prediction outside training data range (extrapolation)</div>', unsafe_allow_html=True)

    st.write("")

    # Binder pie chart
    if binder_total > 0:
        labels  = [l for l, v in [("Fly Ash", fly_ash), ("Slag", slag), ("Metakaolin", metakaoline)] if v > 0]
        values  = [v for v in [fly_ash, slag, metakaoline] if v > 0]
        colors_ = ["#4ECDC4", "#FF6B6B", "#FFE66D"][:len(values)]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors_,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Binder Composition", fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    # PDF download
    report_inputs = {
        "Molarity (M)":          molarity,
        "Fine Aggregate (kg/m³)": fine_agg,
        "Nano Silica (kg/m³)":    nano_silica,
        "NaOH (kg/m³)":           naoh,
        "Na₂SiO₃ (kg/m³)":       na2sio3,
        "Na₂SiO₃/NaOH":          f"{ratio:.3f}",
        "Alk : Binder":           alk_binder,
        "Curing (°C)":            curing,
        "Fly Ash (kg/m³)":        fly_ash,
        "Slag (kg/m³)":           slag,
        "Metakaolin (kg/m³)":     metakaoline,
    }
    pdf_file = make_pdf(prediction, report_inputs, ratio, binder_total)
    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=pdf_file,
        file_name="Geopolymer_Strength_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
