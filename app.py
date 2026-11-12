import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Heat Exchanger Digital Twin",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ AI-Based Heat Exchanger Digital Twin")
st.caption("Thermal Performance Monitoring & Intelligent Fault Detection")

CP = 4180          # J/kg.K (water)
AREA = 12.0        # m2
U_CLEAN = 800.0    # W/m2.K

# ==========================================
# 2. HEAT EXCHANGER MATHEMATICAL MODEL
# ==========================================
def simulate(mh, mc, Thi, Tci, Rf, sensor_fault=False):
    Ch = mh * CP
    Cc = mc * CP
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    Cr = Cmin / Cmax

    Uf = 1 / (1 / U_CLEAN + Rf)
    UA = Uf * AREA
    NTU = UA / Cmin

    if abs(Cr - 1) < 1e-6:
        epsilon = NTU / (1 + NTU)
    else:
        exp_term = np.exp(-NTU * (1 - Cr))
        epsilon = (1 - exp_term) / (1 - Cr * exp_term)

    Q = epsilon * Cmin * (Thi - Tci)
    Tho = Thi - Q / Ch
    Tco = Tci + Q / Cc

    Tho_measured = Tho + (8.0 if sensor_fault else 0.0)
    Tco_measured = Tco

    Qh = Ch * (Thi - Tho_measured)
    Qc = Cc * (Tco_measured - Tci)
    Qavg = (Qh + Qc) / 2
    energy_error = abs(Qh - Qc) / max(abs(Qavg), 1)

    dT1 = Thi - Tco_measured
    dT2 = Tho_measured - Tci

    if dT1 > 0 and dT2 > 0:
        LMTD = dT1 if abs(dT1 - dT2) < 1e-8 else (dT1 - dT2) / np.log(dT1 / dT2)
    else:
        LMTD = 1.0

    U_est = Qavg / (AREA * LMTD)
    effectiveness_est = Qavg / max(Cmin * (Thi - Tci), 1)

    return {
        "Q": Q,
        "Th_out": Tho_measured,
        "Tc_out": Tco_measured,
        "U": U_est,
        "Effectiveness": effectiveness_est,
        "Energy_Error": energy_error,
        "Fouling_U": Uf
    }

FEATURES = ["Hot_Flow", "Cold_Flow", "Heat_Duty", "U", "Effectiveness", "Energy_Error", "Hot_Outlet", "Cold_Outlet"]

def extract_features(mh, mc, Thi, Tci, Rf, sensor_fault=False):
    result = simulate(mh, mc, Thi, Tci, Rf, sensor_fault)
    return {
        "Hot_Flow": mh,
        "Cold_Flow": mc,
        "Heat_Duty": result["Q"],
        "U": result["U"],
        "Effectiveness": result["Effectiveness"],
        "Energy_Error": result["Energy_Error"],
        "Hot_Outlet": result["Th_out"],
        "Cold_Outlet": result["Tc_out"]
    }

@st.cache_resource
def train_ai():
    np.random.seed(42)
    data = []
    for i in range(1500):
        mh = np.random.uniform(0.8, 1.2)
        mc = np.random.uniform(0.8, 1.2)
        Thi = np.random.uniform(80, 100)
        Tci = np.random.uniform(20, 30)
        Rf = np.random.uniform(0.00005, 0.00015)
        row = extract_features(mh, mc, Thi, Tci, Rf)
        data.append(row)
    df = pd.DataFrame(data)
    model = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
    model.fit(df[FEATURES])
    return model

model = train_ai()

# ==========================================
# 3. SIDEBAR & SIMULATION CONTROLS
# ==========================================
st.sidebar.header("Process Conditions")
scenario = st.sidebar.selectbox("Operating Scenario", ["Normal", "Fouling", "Hot Flow Reduction", "Cold Flow Reduction", "Sensor Failure"])

Thi = st.sidebar.slider("Hot Inlet Temperature (°C)", 70.0, 110.0, 90.0)
Tci = st.sidebar.slider("Cold Inlet Temperature (°C)", 10.0, 40.0, 25.0)
mh = st.sidebar.slider("Hot Mass Flow Rate (kg/s)", 0.2, 2.0, 1.0)
mc = st.sidebar.slider("Cold Mass Flow Rate (kg/s)", 0.2, 2.0, 1.2)

Rf = 0.0001
sensor_fault = False

if scenario == "Fouling":
    Rf = 0.0015
elif scenario == "Hot Flow Reduction":
    mh = mh * 0.45
elif scenario == "Cold Flow Reduction":
    mc = mc * 0.45
elif scenario == "Sensor Failure":
    sensor_fault = True

result = simulate(mh, mc, Thi, Tci, Rf, sensor_fault)
features = extract_features(mh, mc, Thi, Tci, Rf, sensor_fault)
X = pd.DataFrame([features], columns=FEATURES)
prediction = model.predict(X)[0]
is_anomaly = prediction == -1

# ==========================================
# 4. DASHBOARD DISPLAY
# ==========================================
st.header("1. Thermal Performance Monitoring")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Heat Duty", f"{result['Q']/1000:.2f} kW")
c2.metric("Estimated Overall U", f"{result['U']:.1f} W/m².K")
c3.metric("Effectiveness", f"{result['Effectiveness']:.3f}")
c4.metric("Energy Imbalance", f"{features['Energy_Error']*100:.2f}%")

st.header("2. AI Fault Detection")
if is_anomaly:
    st.error("⚠️ AI RESULT: ANOMALY DETECTED")
else:
    st.success("✅ AI RESULT: NORMAL OPERATION")# ==========================================
# 5. PLANT DATA VALIDATION (REAL PHE DATA)
# ==========================================
import plant_data as pdx

st.header("3. Plant Data Validation (Real PHE Data)")
up = st.file_uploader("Upload PHE plant data Excel file (.xlsx)", type="xlsx")

if up is not None:
    raw = pdx.load_raw(up)
    unit = st.radio("Select Equipment", ["P1", "P2"], horizontal=True,
                    format_func=lambda x: "PHE#1" if x == "P1" else "PHE#2")

    flags = pdx.quality_flags(raw, unit)
    clean = ~flags["any_flag"]
    n_train = len(raw) // 2

    par, val, pred = pdx.validate(raw, unit, n_train, clean)

    k1, k2, k3 = st.columns(3)
    k1.metric("Calibrated UA", f"{par['UA']:.0f} W/K")
    k2.metric("Implied Lye Flow", f"{par['V_lye_implied']:.2f} m³/h", f"assumed 7.0 m³/h", delta_color="off")
    k3.metric("Data Quality Flags", int(flags["any_flag"].sum()))

    st.subheader("Hold-out Test (Unseen 2nd Half)")
    st.dataframe(val.round(3), use_container_width=True)

    fig_val = go.Figure()
    for col, name in [("Lye_meas", "Lye outlet measured"), ("Lye_pred", "Lye outlet predicted"),
                      ("Wtr_meas", "Water outlet measured"), ("Wtr_pred", "Water outlet predicted")]:
        fig_val.add_trace(go.Scatter(x=pred["Time"], y=pred[col], name=name,
                                     mode="lines" if "pred" in col else "markers"))
    fig_val.update_layout(title="Digital Twin vs Plant Measurement", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_val, use_container_width=True)

    st.subheader("AI Anomaly Detection on Real Plant Data")
    m = pdx.detect(pdx.features(raw, unit, par), n_train, clean)
    fig_ai = px.line(m, x="Time", y="score", title="Isolation Forest Anomaly Score (Lower = More Anomalous)")
    an = m[m["AI_anomaly"]]
    fig_ai.add_scatter(x=an["Time"], y=an["score"], mode="markers", name="AI Anomaly", marker=dict(size=10, color="red"))
    for t in pdx.MANUAL_EVENTS:
        fig_ai.add_vline(x=t, line_dash="dash", line_color="orange")
    st.plotly_chart(fig_ai, use_container_width=True)

    st.write("Comparison: Engineer/Rule Flags vs AI Flags")
    st.dataframe(pdx.compare_with_engineer(m, flags))
    st.caption("Flow rates are assumed (lye 7, water 10 m³/h). Temperature-based metrics are flow-independent.")
    # ==========================================
# 6. EXECUTIVE SUMMARY & PLANT INSIGHTS
# ==========================================
if up is not None:
    st.markdown("---")
    st.header("4. Engineering Insights & Plant Summary")
    
    # ดึงข้อมูลสรุปมาแสดงเป็นตารางสวยงาม
    summary_table = pdx.get_plant_summary(m, flags, unit)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📊 Plant Health Overview")
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
    
    with col_b:
        st.subheader("💡 Key Engineering Takeaways")
        st.info(
            "• **Data Integrity:** Physical consistency checks successfully flagged "
            "reversing pressure drops and temperature limit violations.\n\n"
            "• **Model Calibration:** Using nonlinear least squares, the calibrated UA "
            "reflects actual thermal performance under assumed flow conditions.\n\n"
            "• **AI Performance:** Isolation Forest effectively isolates operational "
            "outliers without requiring pre-labeled failure data."
        )
    
    # เพิ่มปุ่มฟิลเตอร์ดูเฉพาะจุดที่ AI จับความผิดปกติ
    if st.checkbox("🔍 Filter and show only AI-detected anomalies in history"):
        st.subheader("Filtered Anomalous Records")
        anomalous_records = m[m["AI_anomaly"]][["Time", "score", "LMTD", "dP_wtr", "dP_lye"]]
        st.dataframe(anomalous_records, use_container_width=True)