import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Geopolymer Mortar Strength Predictor",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.main-header {
    background: linear-gradient(90deg, #6a00f4, #00b4d8, #00d4a6);
    padding: 26px 30px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
}
.main-header h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 800;
}
.main-header p {
    margin: 6px 0 0 0;
    font-size: 18px;
}
.disclaimer {
    margin-top: 10px;
    font-size: 14px;
    color: #f1f3f5;
}
.card {
    background: #f8f9ff;
    border: 1px solid #d9def2;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}
.result-card {
    background: linear-gradient(135deg, #0ead69, #1b9aaa);
    color: white;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    margin-top: 12px;
}
.result-card h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
}
.result-card h1 {
    margin: 8px 0 0 0;
    font-size: 36px;
    font-weight: 900;
}
.check-good {
    padding: 10px 12px;
    background: #e8fff1;
    border-left: 5px solid #18a957;
    border-radius: 8px;
    margin-bottom: 10px;
}
.check-warn {
    padding: 10px 12px;
    background: #fff6e5;
    border-left: 5px solid #d98e04;
    border-radius: 8px;
    margin-bottom: 10px;
}
.check-bad {
    padding: 10px 12px;
    background: #fff0f0;
    border-left: 5px solid #d62828;
    border-radius: 8px;
    margin-bottom: 10px;
}
.small-text {
    font-size: 14px;
    color: #4f5d75;
}
.note-box {
    padding: 12px;
    background: #eef6ff;
    border-left: 5px solid #3282b8;
    border-radius: 8px;
    margin-top: 10px;
}
.range-title {
    color: #0f4c81;
    font-weight: 800;
    font-size: 28px;
    line-height: 1.2;
    margin-bottom: 18px;
}
.range-label {
    color: #233044;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 0px;
}
.range-value {
    color: #3d4656;
    margin-top: 0px;
    margin-bottom: 8px;
}
.gauge-high {
    padding: 12px;
    background: #e8fff1;
    border-radius: 10px;
    font-weight: 700;
    text-align: center;
    color: #127a43;
}
.gauge-medium {
    padding: 12px;
    background: #fff6e5;
    border-radius: 10px;
    font-weight: 700;
    text-align: center;
    color: #a86a00;
}
.gauge-low {
    padding: 12px;
    background: #fff0f0;
    border-radius: 10px;
    font-weight: 700;
    text-align: center;
    color: #b42318;
}
div.stButton > button:first-child {
    background: linear-gradient(135deg,#ff7b00,#ff006e);
    color: white;
    font-weight: 700;
    font-size: 18px;
    border-radius: 12px;
    height: 55px;
    border: none;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg,#ff006e,#ff7b00);
    color: white;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h3 {
    color: #0f4c81;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("catboost_model.pkl")

metrics_df = pd.read_excel("Final_Model_Metrics_Comparison.xlsx")
metrics_10f = dict(
    zip(metrics_df["Metric"], metrics_df["10-Fold CV (Pooled)"]))
metrics_mc = dict(
    zip(metrics_df["Metric"], metrics_df["Monte Carlo (100x Avg)"]))

dataset_ranges = {
    "Molarity (M)": (3.0, 16.0),
    "Fine Aggregate (kg/m³)": (432.0, 1500.0),
    "Nano Silica (kg/m³)": (0.0, 50.0),
    "NaOH (kg/m³)": (64.28, 130.0),
    "Na2SiO3 (kg/m³)": (127.27, 250.0),
    "Na2SiO3/NaOH": (1.5, 2.5),
    "Alk: Binder": (0.30, 0.70),
    "Curing Condition": (20.0, 70.0),
    "Fly ash (kg/m³)": (0.0, 734.0),
    "Slag (kg/m³)": (71.42, 650.0),
    "Metakaoline (kg/m³)": (225.0, 450.0),
}


def make_pdf(prediction, inputs, ratio, binder_total, m10f, mmc, confidence_text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#2B2D42"))
    pdf.rect(0, height - 90, width, 90, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 45,
                   "Geopolymer Mortar Strength Prediction Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, height - 65, "For scientific and research use only")

    y = height - 120
    pdf.setFillColor(colors.black)

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Predicted Output")
    y -= 25

    pdf.setFont("Helvetica-Bold", 18)
    pdf.setFillColor(colors.HexColor("#0A9396"))
    pdf.drawString(
        40, y, f"28-Day Compressive Strength = {prediction:.2f} MPa")
    y -= 25

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y, f"Prediction reliability: {confidence_text}")
    y -= 35

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Input Parameters")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for key, value in inputs.items():
        pdf.drawString(45, y, f"{key}: {value}")
        y -= 18
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)

    y -= 8
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, f"Calculated Na2SiO3/NaOH Ratio: {ratio:.3f}")
    y -= 20
    pdf.drawString(40, y, f"Total Binder Content: {binder_total:.2f} kg/m³")

    y -= 35
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Validation Performance")
    y -= 22

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(45, y, "10-Fold Cross-Validation")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        55, y, f"R² = {m10f['R2']:.3f} | RMSE = {m10f['RMSE']:.3f} | MAPE = {m10f['MAPE (%)']:.2f}")
    y -= 18
    pdf.drawString(
        55, y, f"r = {m10f['Pearson (r)']:.3f} | IoA = {m10f['IoA']:.3f}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(45, y, "Monte Carlo Cross-Validation (100 Repeats)")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        55, y, f"R² = {mmc['R2']:.3f} | RMSE = {mmc['RMSE']:.3f} | MAPE = {mmc['MAPE (%)']:.2f}")
    y -= 18
    pdf.drawString(
        55, y, f"r = {mmc['Pearson (r)']:.3f} | IoA = {mmc['IoA']:.3f}")
    y -= 24

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(
        40, y, "Prediction is generated by the final CatBoost model trained on the full dataset.")
    y -= 15
    pdf.drawString(
        40, y, "10-Fold and Monte Carlo values are validation statistics.")
    pdf.save()
    buffer.seek(0)
    return buffer


def show_check(message, level="good"):
    css = {
        "good": "check-good",
        "warn": "check-warn",
        "bad": "check-bad"
    }[level]
    st.markdown(f'<div class="{css}">{message}</div>', unsafe_allow_html=True)


def confidence_level(prediction, outside_count):
    if outside_count == 0 and 10 <= prediction <= 120:
        return "High"
    if outside_count <= 2 and 5 <= prediction <= 140:
        return "Moderate"
    return "Low"


def render_gauge(level):
    if level == "High":
        st.markdown(
            '<div class="gauge-high">Prediction Reliability: High</div>', unsafe_allow_html=True)
    elif level == "Moderate":
        st.markdown(
            '<div class="gauge-medium">Prediction Reliability: Moderate</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="gauge-low">Prediction Reliability: Low</div>', unsafe_allow_html=True)


def out_of_range_checks(values_dict):
    messages = []
    outside = 0
    for name, value in values_dict.items():
        min_v, max_v = dataset_ranges[name]
        if value < min_v or value > max_v:
            outside += 1
            messages.append(
                f"{name} is outside the dataset range ({min_v} – {max_v}).")
    return outside, messages


st.markdown("""
<div class="main-header">
    <h1>Geopolymer Mortar Strength Predictor</h1>
    <p>Interactive machine-learning tool based on a trained CatBoost model</p>
    <div class="disclaimer">For scientific and research use only</div>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.markdown("""
<div style="
border-left:6px solid #0f4c81;
padding-left:12px;
font-size:18px;
font-weight:800;
line-height:1.25;
color:#0f4c81;
">
Dataset Input Ranges Used for Model Training and Validation
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="range-label">Molarity (M)</div><div class="range-value">3 – 16</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Fine Aggregate (kg/m³)</div><div class="range-value">432 – 1500</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Nano Silica (kg/m³)</div><div class="range-value">0 – 50</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">NaOH (kg/m³)</div><div class="range-value">64.28 – 130</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Na2SiO3 (kg/m³)</div><div class="range-value">127.27 – 250</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Na2SiO3/NaOH</div><div class="range-value">1.5 – 2.5</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Alk: Binder</div><div class="range-value">0.3 – 0.7</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Curing Condition</div><div class="range-value">20 – 70</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Binder")
    st.markdown('<div class="range-label">Fly ash (kg/m³)</div><div class="range-value">0 – 734</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Slag (kg/m³)</div><div class="range-value">71.42 – 650</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="range-label">Metakaoline (kg/m³)</div><div class="range-value">225 – 450</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Use values within dataset ranges for more reliable prediction.")

left, right = st.columns([2.2, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Mixture Design Inputs")

    st.markdown("### Main Parameters")
    molarity = st.slider("Molarity (M)", min_value=3.0,
                         max_value=16.0, value=10.0, step=0.1)
    fine_agg = st.slider("Fine Aggregate (kg/m³)", min_value=432.0,
                         max_value=1500.0, value=700.0, step=1.0)
    nano_silica = st.slider("Nano Silica (kg/m³)",
                            min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    naoh = st.slider("NaOH (kg/m³)", min_value=64.28,
                     max_value=130.0, value=80.0, step=0.01)
    na2sio3 = st.slider("Na2SiO3 (kg/m³)", min_value=127.27,
                        max_value=250.0, value=160.0, step=0.01)
    alk_binder = st.slider("Alk: Binder", min_value=0.30,
                           max_value=0.70, value=0.45, step=0.01)
    curing_condition = st.slider(
        "Curing Condition", min_value=20.0, max_value=70.0, value=28.0, step=1.0)

    st.markdown("### Binder")
    b1, b2, b3 = st.columns(3)

    with b1:
        fly_ash = st.slider("Fly ash (kg/m³)", min_value=0.0,
                            max_value=734.0, value=200.0, step=1.0)

    with b2:
        slag = st.slider("Slag (kg/m³)", min_value=71.42,
                         max_value=650.0, value=100.0, step=0.01)

    with b3:
        metakaoline = st.slider(
            "Metakaoline (kg/m³)", min_value=225.0, max_value=450.0, value=250.0, step=1.0)

    predict = st.button("Calculate Compressive Strength",
                        use_container_width=True)
    result_placeholder = st.container()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Machine Learning Model")

    st.markdown("""
**Algorithm:** CatBoost Regressor  

**Dataset size:** 75 experimental geopolymer mortar mixes  

**Input variables:** 11 mixture design parameters  

**Target variable:** 28-day compressive strength (MPa)  

**Validation methods:**  
• 10-Fold Cross-Validation  
• Monte Carlo Cross-Validation (100 repetitions)
""")

    st.markdown("""
    <div class="note-box">
    The predictor uses the <b>final CatBoost model trained on the full dataset</b>.
    The 10-Fold and Monte Carlo values shown below are <b>validation results</b> used to evaluate model performance.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Validation Performance")

    st.markdown("**10-Fold Cross-Validation**")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("R²", f"{metrics_10f['R2']:.3f}")
    with a2:
        st.metric("RMSE", f"{metrics_10f['RMSE']:.3f}")
    with a3:
        st.metric("MAPE (%)", f"{metrics_10f['MAPE (%)']:.2f}")

    a4, a5 = st.columns(2)
    with a4:
        st.metric("r", f"{metrics_10f['Pearson (r)']:.3f}")
    with a5:
        st.metric("IoA", f"{metrics_10f['IoA']:.3f}")

    st.markdown("---")

    st.markdown("**Monte Carlo Cross-Validation (100 Repeats)**")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R² ", f"{metrics_mc['R2']:.3f}")
    with m2:
        st.metric("RMSE ", f"{metrics_mc['RMSE']:.3f}")
    with m3:
        st.metric("MAPE (%) ", f"{metrics_mc['MAPE (%)']:.2f}")

    m4, m5 = st.columns(2)
    with m4:
        st.metric("r ", f"{metrics_mc['Pearson (r)']:.3f}")
    with m5:
        st.metric("IoA ", f"{metrics_mc['IoA']:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Mixture Checks")

    ratio_live = na2sio3 / naoh if naoh != 0 else 0
    binder_total_live = fly_ash + slag + metakaoline

    st.metric("Na2SiO3 / NaOH", f"{ratio_live:.3f}")
    st.metric("Total Binder", f"{binder_total_live:.2f} kg/m³")

    current_values = {
        "Molarity (M)": molarity,
        "Fine Aggregate (kg/m³)": fine_agg,
        "Nano Silica (kg/m³)": nano_silica,
        "NaOH (kg/m³)": naoh,
        "Na2SiO3 (kg/m³)": na2sio3,
        "Na2SiO3/NaOH": ratio_live,
        "Alk: Binder": alk_binder,
        "Curing Condition": curing_condition,
        "Fly ash (kg/m³)": fly_ash,
        "Slag (kg/m³)": slag,
        "Metakaoline (kg/m³)": metakaoline
    }

    outside_count_live, out_msgs_live = out_of_range_checks(current_values)

    if 1.5 <= ratio_live <= 2.5:
        show_check("Na2SiO3/NaOH ratio is within the dataset range.", "good")
    elif 1.3 <= ratio_live <= 2.7:
        show_check("Na2SiO3/NaOH ratio is near the dataset boundary.", "warn")
    else:
        show_check("Na2SiO3/NaOH ratio is outside the dataset range.", "bad")

    if binder_total_live > 0:
        show_check("Binder total is valid.", "good")
    else:
        show_check("Binder total must be greater than zero.", "bad")

    if outside_count_live == 0:
        show_check(
            "All inputs are within the dataset range used for training and validation.", "good")
    else:
        for msg in out_msgs_live:
            show_check(msg, "warn")

    st.markdown('<div class="small-text">Prediction is more reliable when the entered values remain close to the original experimental dataset.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if predict:
    with result_placeholder:
        ratio = na2sio3 / naoh if naoh != 0 else 0
        binder_total = fly_ash + slag + metakaoline

        input_data = pd.DataFrame([{
            "Molarity (M)": molarity,
            "Fine Aggregate": fine_agg,
            "Nano Silica (Kg)": nano_silica,
            "NaOH": naoh,
            "Na2SiO3": na2sio3,
            "Na2SiO3/NaOH": ratio,
            "Alk: Binder": alk_binder,
            "Curing Condition": curing_condition,
            "Fly ash": fly_ash,
            "Slag": slag,
            "Metakaoline": metakaoline
        }])

        prediction = model.predict(input_data)[0]

        current_values_pred = {
            "Molarity (M)": molarity,
            "Fine Aggregate (kg/m³)": fine_agg,
            "Nano Silica (kg/m³)": nano_silica,
            "NaOH (kg/m³)": naoh,
            "Na2SiO3 (kg/m³)": na2sio3,
            "Na2SiO3/NaOH": ratio,
            "Alk: Binder": alk_binder,
            "Curing Condition": curing_condition,
            "Fly ash (kg/m³)": fly_ash,
            "Slag (kg/m³)": slag,
            "Metakaoline (kg/m³)": metakaoline
        }

        outside_count_pred, out_msgs_pred = out_of_range_checks(
            current_values_pred)
        conf = confidence_level(prediction, outside_count_pred)

        st.markdown(f"""
        <div class="result-card">
            <h2>Predicted 28-Day Compressive Strength</h2>
            <h1>{prediction:.2f} MPa</h1>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            render_gauge(conf)
        with c2:
            st.metric("Na2SiO3/NaOH Ratio", f"{ratio:.3f}")

        if outside_count_pred > 0:
            st.markdown("### Input Validity Warning")
            for msg in out_msgs_pred:
                show_check(msg, "warn")

        st.write("")
        st.subheader("Binder Distribution")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [fly_ash, slag, metakaoline],
            labels=["Fly ash", "Slag", "Metakaoline"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

        report_inputs = {
            "Molarity (M)": molarity,
            "Fine Aggregate (kg/m³)": fine_agg,
            "Nano Silica (kg/m³)": nano_silica,
            "NaOH (kg/m³)": naoh,
            "Na2SiO3 (kg/m³)": na2sio3,
            "Alk: Binder": alk_binder,
            "Curing Condition": curing_condition,
            "Fly ash (kg/m³)": fly_ash,
            "Slag (kg/m³)": slag,
            "Metakaoline (kg/m³)": metakaoline
        }

        pdf_file = make_pdf(
            prediction,
            report_inputs,
            ratio,
            binder_total,
            metrics_10f,
            metrics_mc,
            conf
        )

        st.download_button(
            label="Download Prediction Report (PDF)",
            data=pdf_file,
            file_name="Geopolymer_Mortar_Prediction_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
