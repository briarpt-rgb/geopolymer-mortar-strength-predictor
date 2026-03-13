import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & STYLE (EXACT ORIGINAL)
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
.check-bad { padding: 10px; background: #fff0f0; border-left: 5px solid #d62828; border-radius: 8px; margin-bottom: 10px; }
.gauge-high { padding: 12px; background: #e8fff1; border-radius: 10px; font-weight: 700; text-align: center; color: #127a43; }
.gauge-low { padding: 12px; background: #fff0f0; border-radius: 10px; font-weight: 700; text-align: center; color: #b42318; }
div.stButton > button:first-child {
    background: linear-gradient(135deg,#ff7b00,#ff006e);
    color: white; font-weight: 700; font-size: 18px;
    border-radius: 12px; height: 55px; border: none;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING (FIXED)
# ==========================================
model = joblib.load("catboost_model.pkl")

# Hardcoded from your latest correct notebook run
metrics_10f = {"R2": 0.835, "RMSE": 7.175,
               "MAPE (%)": 13.94, "r": 0.914, "IoA": 0.952}
metrics_mc = {"R2": 0.783, "RMSE": 7.770,
              "MAPE (%)": 16.38, "r": 0.885, "IoA": 0.935}

dataset_ranges = {
    "Molarity (M)": (3.0, 16.0),
    "Fine Aggregate (kg/m³)": (432.0, 1500.0),
    "Nano Silica (kg/m³)": (0.0, 50.0),
    "NaOH (kg/m³)": (64.28, 130.0),
    "Na2SIO3 (kg/m³)": (127.27, 250.0),
    "Na2SiO3/NaOH": (1.5, 2.5),
    "Alk: Binder": (0.30, 0.70),
    "Curing Condition": (20.0, 70.0),
    "Fly ash (kg/m³)": (0.0, 734.0),
    "Slag (kg/m³)": (0.0, 650.0),
    "Metakaoline (kg/m³)": (0.0, 450.0),
}

# ==========================================
# 3. HELPER FUNCTIONS (PDF & CHECKS)
# ==========================================


def make_pdf(prediction, inputs, ratio, binder_total):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 800, "Geopolymer Mortar Strength Prediction Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 775, f"28-Day Strength: {prediction:.2f} MPa")
    pdf.drawString(50, 755, f"Na2SiO3/NaOH Ratio: {ratio:.3f}")
    pdf.drawString(50, 735, f"Total Binder: {binder_total:.2f} kg/m³")
    y = 700
    for k, v in inputs.items():
        pdf.drawString(50, y, f"{k}: {v}")
        y -= 15
    pdf.save()
    buffer.seek(0)
    return buffer


def show_check(msg, level):
    st.markdown(
        f'<div class="check-{level}">{msg}</div>', unsafe_allow_html=True)


# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>Geopolymer Mortar Strength Predictor</h1>
    <p>Interactive machine-learning tool based on a trained CatBoost model</p>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.header("Dataset Input Ranges")
    for k, (min_v, max_v) in dataset_ranges.items():
        st.markdown(f"**{k}**: {min_v} – {max_v}")

left, right = st.columns([2.2, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Mixture Design Inputs")
    molarity = st.slider("Molarity (M)", 3.0, 16.0, 10.0)
    fine_agg = st.slider("Fine Aggregate (kg/m³)", 432.0, 1500.0, 700.0)
    nano_silica = st.slider("Nano Silica (kg/m³)", 0.0, 50.0, 5.0)
    naoh = st.slider("NaOH (kg/m³)", 64.28, 130.0, 80.0)
    na2sio3 = st.slider("Na2SIO3 (kg/m³)", 127.27, 250.0, 160.0)
    alk_binder = st.slider("Alk: Binder", 0.30, 0.70, 0.45)
    curing = st.slider("Curing Condition", 20.0, 70.0, 28.0)

    st.markdown("### Binder")
    b1, b2, b3 = st.columns(3)
    fly_ash = b1.number_input("Fly ash (kg/m³)", 0.0, 734.0, 200.0)
    slag = b2.number_input("Slag (kg/m³)", 0.0, 650.0, 100.0)
    metakaoline = b3.number_input("Metakaoline (kg/m³)", 0.0, 450.0, 250.0)

    predict = st.button("Calculate Compressive Strength",
                        use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    st.metric("10-Fold R²", f"{metrics_10f['R2']}")
    st.metric("Monte Carlo R²", f"{metrics_mc['R2']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Mixture Checks")
    ratio = na2sio3 / naoh if naoh != 0 else 0
    binder_total = fly_ash + slag + metakaoline
    st.metric("Na2SiO3 / NaOH", f"{ratio:.3f}")
    st.metric("Total Binder", f"{binder_total:.2f} kg/m³")

    if 1.5 <= ratio <= 2.5:
        show_check("Ratio within range", "good")
    else:
        show_check("Ratio out of range", "warn")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if predict:
    # Feature order must match your notebook exactly
    input_data = pd.DataFrame([{
        "Molarity (M)": molarity,
        "Fly ash": fly_ash,
        "Slag": slag,
        "Metakaoline": metakaoline,
        "Fine Aggregate": fine_agg,
        "Nano Silica (Kg)": nano_silica,
        "NaOH": naoh,
        "Na2SIO3": na2sio3,
        "Na2SiO3/NaOH": ratio,
        "Alk: Binder": alk_binder,
        "Curing Condition": curing
    }])

    prediction = model.predict(input_data)[0]

    st.markdown(
        f'<div class="result-card"><h2>Predicted 28-Day Strength</h2><h1>{prediction:.2f} MPa</h1></div>', unsafe_allow_html=True)

    st.write("")
    if 10 <= prediction <= 120:
        st.markdown('<div class="gauge-high">Reliability: High</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="gauge-low">Reliability: Low</div>',
                    unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([fly_ash, slag, metakaoline], labels=[
           "Fly ash", "Slag", "Metakaoline"], autopct="%1.1f%%")
    st.pyplot(fig)

    # RESTORED DOWNLOAD BUTTON
    report_inputs = {"Molarity": molarity,
                     "Fine Aggregate": fine_agg, "NaOH": naoh, "Na2SIO3": na2sio3}
    pdf_file = make_pdf(prediction, report_inputs, ratio, binder_total)
    st.download_button(label="Download Prediction Report (PDF)", data=pdf_file,
                       file_name="Prediction_Report.pdf", mime="application/pdf", use_container_width=True)
