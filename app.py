import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI-Based Heat Exchanger Digital Twin", page_icon="🌡️", layout="wide")

RHO_L, CP_L, RHO_W, CP_W = 1450.0, 3200.0, 996.0, 4180.0  # NaOH 43% / น้ำหล่อเย็น
DEFAULT_FILE = "PHE_Performance_Analysis_v3.xlsx"           # วางไว้ใน repo เพื่อโหลดอัตโนมัติ
PHES = ["PHE#1", "PHE#2"]
FEATS_IF = ["th_i", "tc_i", "th_o", "eps"]

# (ΔT lye in, ΔT water in, ตัวคูณ lye flow, ตัวคูณ water flow, UA loss %)
SCEN = {
    "Normal Operation": (0, 0, 1.0, 1.0, 0),
    "Fouling Buildup (UA -30%)": (0, 0, 1.0, 1.0, 30),
    "Severe Fouling (UA -60%)": (0, 0, 1.0, 1.0, 60),
    "Lye Flow Drop (-50%)": (0, 0, 0.5, 1.0, 0),
    "Cooling Water Flow Drop (-50%)": (0, 0, 1.0, 0.5, 0),
    "Cooling Water Temp Spike (+5°C)": (0, 5, 1.0, 1.0, 0),
    "Hot Lye Surge (+5°C)": (5, 0, 1.0, 1.0, 0),
}


# ============================================================
# Physics: ε-NTU (counter-flow) และสมการสมรรถนะ
# ============================================================
def caps(vl, vw):
    """heat capacity rate [W/K] จากอัตราการไหลเชิงปริมาตร [m3/hr]"""
    return vl / 3600 * RHO_L * CP_L, vw / 3600 * RHO_W * CP_W


def lmtd(th_i, th_o, tc_i, tc_o):
    d1 = np.asarray(th_i - tc_o, float)
    d2 = np.asarray(th_o - tc_i, float)
    with np.errstate(all="ignore"):
        out = np.where(np.abs(d1 - d2) < 1e-6, d1, (d1 - d2) / np.log(d1 / d2))
    return np.where((d1 > 0) & (d2 > 0), out, np.nan)


def calc(th_i, th_o, tc_i, tc_o, vl, vw):
    cl, cw = caps(vl, vw)
    ql = cl * (th_i - th_o) / 1000
    qw = cw * (tc_o - tc_i) / 1000
    q = (ql + qw) / 2
    L = lmtd(th_i, th_o, tc_i, tc_o)
    with np.errstate(all="ignore"):
        ua = ql * 1000 / L                      # UA และ ε อิงฝั่ง lye (ไม่ขึ้นกับ flow น้ำที่สมมติ)
        eps = ql * 1000 / (min(cl, cw) * (th_i - tc_i))
    return dict(Ql=ql, Qw=qw, Q=q, LMTD=L, UA=ua, eps=eps)


def predict(th_i, tc_i, ua, vl, vw):
    """ทำนายอุณหภูมิขาออกจาก UA + อัตราการไหล (ε-NTU counter-flow)"""
    cl, cw = caps(vl, vw)
    cmin, cmax = min(cl, cw), max(cl, cw)
    cr = cmin / cmax
    ntu = ua / cmin
    e = np.exp(-ntu * (1 - cr))
    eps = ntu / (1 + ntu) if abs(1 - cr) < 1e-9 else (1 - e) / (1 - cr * e)
    q = eps * cmin * (th_i - tc_i)
    return th_i - q / cl, tc_i + q / cw, q / 1000, eps, ntu


# ============================================================
# Data: โหลด / ทำความสะอาด / คำนวณ
# ============================================================
@st.cache_data(show_spinner=False)
def load_plant(blob: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(blob), sheet_name="Raw Data", header=None)
    d = raw.iloc[4:, :17].copy()
    names = ["Time"]
    for kind in ("T", "P"):
        for p in (1, 2):
            names += [f"{kind}{p}_{s}" for s in ("wi", "wo", "li", "lo")]
    d.columns = names
    d["Time"] = pd.to_datetime(d["Time"], format="%d %b %y %H:%M", errors="coerce")
    for c in names[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    tcols = [c for c in names if c[0] == "T" and c != "Time"]
    d = d.dropna(subset=["Time"]).dropna(subset=tcols, how="all")  # ตัดแถว placeholder
    return d.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def derive(d: pd.DataFrame, vl: float, vw: float) -> pd.DataFrame:
    parts = []
    cmin = min(caps(vl, vw))
    for p in (1, 2):
        s = pd.DataFrame({
            "Time": d["Time"], "phe": f"PHE#{p}",
            "th_i": d[f"T{p}_li"], "th_o": d[f"T{p}_lo"],
            "tc_i": d[f"T{p}_wi"], "tc_o": d[f"T{p}_wo"],
            "dP_w": d[f"P{p}_wi"] - d[f"P{p}_wo"], "dP_l": d[f"P{p}_li"] - d[f"P{p}_lo"],
        })
        s["inv"] = (s.dP_w < 0) | (s.dP_l < 0)           # ความดัน outlet > inlet (น่าสงสัย)
        s.loc[s.dP_w < 0, "dP_w"] = np.nan
        s.loc[s.dP_l < 0, "dP_l"] = np.nan
        for k, v in calc(s.th_i.values, s.th_o.values, s.tc_i.values, s.tc_o.values, vl, vw).items():
            s[k] = v
        s["NTU"] = s.UA / cmin
        s["bal"] = (s.Ql - s.Qw).abs() / s.Q * 100
        parts.append(s)
    out = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    out["day"] = (out.Time - out.Time.min()).dt.total_seconds() / 86400
    return out


@st.cache_data(show_spinner="กำลังเทรนโมเดลจากข้อมูลจริง...")
def train(D: pd.DataFrame, phe: str, vl: float, vw: float) -> dict:
    s = D[D.phe == phe].dropna(subset=FEATS_IF + ["Q"])
    s = s[(s.Q > 0) & (s.UA > 0)].reset_index(drop=True)
    X, y = s[["th_i", "tc_i"]].values, s.th_o.values

    # ประเมินโมเดลแบบ time-series CV (ไม่ใช้อนาคตทำนายอดีต)
    res = {"RandomForest (AI)": [], "Physics ε-NTU": [], "Naive (ΔT เฉลี่ย)": []}
    ytrue, rf_err = [], []
    for tr, te in TimeSeriesSplit(5).split(X):
        rf = RandomForestRegressor(200, min_samples_leaf=5, random_state=0).fit(X[tr], y[tr])
        p_rf = rf.predict(X[te])
        p_ph = predict(s.th_i.values[te], s.tc_i.values[te], np.median(s.UA.values[tr]), vl, vw)[0]
        p_nv = s.th_i.values[te] - np.mean(s.th_i.values[tr] - y[tr])
        for k, p in zip(res, (p_rf, p_ph, p_nv)):
            res[k].append(np.abs(p - y[te]))
        rf_err.append(p_rf - y[te])
    metrics = pd.DataFrame({"MAE ของ Lye outlet (°C)": {k: np.concatenate(v).mean() for k, v in res.items()}}).round(3)
    sigma = max(float(np.std(np.concatenate(rf_err))), 0.3)

    rf = RandomForestRegressor(200, min_samples_leaf=5, random_state=0).fit(X, y)
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, contamination=0.05, random_state=0))
    iso.fit(s[FEATS_IF])
    s["score"] = iso.decision_function(s[FEATS_IF])
    s["anom"] = iso.predict(s[FEATS_IF]) == -1
    return dict(
        rf=rf, iso=iso, sigma=sigma, metrics=metrics, scored=s,
        ua_med=float(s.UA.median()), ua_lo=float(s.UA.quantile(.05)), ua_hi=float(s.UA.quantile(.95)),
        rng={c: (float(s[c].min()), float(s[c].max())) for c in ("th_i", "tc_i", "th_o")},
    )


# ============================================================
# Sidebar: แหล่งข้อมูล
# ============================================================
st.title("🌡️ AI-Based Heat Exchanger Digital Twin")
st.caption("Digitalization in Chemical Engineering — Caustic Absorption Plant (PHE#1 / PHE#2, Lye 43% NaOH ↔ Cooling Water)")

with st.sidebar:
    st.header("📁 แหล่งข้อมูล")
    up = st.file_uploader("อัปโหลดไฟล์ PHE_Performance_Analysis (.xlsx)", type=["xlsx"])
    blob = up.getvalue() if up else None  # ต้องอัปโหลดไฟล์ก่อนจึงจะแสดงผล
    with st.expander("⚠️ อัตราการไหลฐาน (ค่าสมมติ)"):
        st.number_input("Lye flow (m³/hr)", 0.5, 50.0, 7.0, 0.5, key="vl0")
        st.number_input("Cooling water flow (m³/hr)", 0.5, 80.0, 10.0, 0.5, key="vw0")
        st.caption("ยังไม่มีค่าจริงจาก pump nameplate — เมื่อได้ค่าจริงให้แก้ตรงนี้ โมเดลทั้งหมดจะคำนวณใหม่")

if blob is None:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูล Plant Data (.xlsx) ที่แถบด้านซ้ายก่อน ระบบจึงจะวิเคราะห์และเปิดตัวจำลองครับ")
    st.stop()
try:
    raw = load_plant(blob)
except Exception as e:
    st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}\n\nต้องมีชีต 'Raw Data' (หัวตารางแถวที่ 4, ข้อมูลเริ่มแถวที่ 5)")
    st.stop()

vl0, vw0 = st.session_state.vl0, st.session_state.vw0
D = derive(raw, vl0, vw0)
M = {p: train(D, p, vl0, vw0) for p in PHES}

# ============================================================
# Sidebar: Simulator Controls (ช่วงอิงข้อมูลจริง + preset + replay)
# ============================================================
st.session_state.bases = {p: (M[p]["scored"].th_i.median(), M[p]["scored"].tc_i.median()) for p in PHES}
st.session_state.hist = D[["Time", "phe", "th_i", "tc_i"]]


def set_vals(th, tc, vl, vw, loss):
    st.session_state.update(th=float(th), tc=float(tc), vl=float(vl), vw=float(vw), loss=float(loss))


def cb_reset():
    b = st.session_state.bases[st.session_state.phe]
    set_vals(b[0], b[1], st.session_state.vl0, st.session_state.vw0, 0)
    st.session_state.scen, st.session_state.replay = "Normal Operation", "— ไม่ใช้ —"


def cb_scen():
    b = st.session_state.bases[st.session_state.phe]
    a, c, fl, fw, loss = SCEN[st.session_state.scen]
    set_vals(b[0] + a, b[1] + c, st.session_state.vl0 * fl, st.session_state.vw0 * fw, loss)


def cb_replay():
    t = st.session_state.replay
    if t == "— ไม่ใช้ —":
        return
    h = st.session_state.hist
    h = h[(h.phe == st.session_state.phe) & (h.Time.dt.strftime("%d %b %H:%M") == t)]
    if len(h):
        set_vals(h.th_i.iloc[0], h.tc_i.iloc[0], st.session_state.vl0, st.session_state.vw0, 0)
        st.session_state.scen = "Normal Operation"


phe_now = st.session_state.get("phe", PHES[0])
rg = M[phe_now]["rng"]
RNG = {
    "th": (float(np.floor(rg["th_i"][0] - 5)), float(np.ceil(rg["th_i"][1] + 5))),
    "tc": (float(np.floor(rg["tc_i"][0] - 5)), float(np.ceil(rg["tc_i"][1] + 5))),
    "vl": (round(0.1 * vl0, 1), round(1.5 * vl0, 1)),
    "vw": (round(0.1 * vw0, 1), round(1.5 * vw0, 1)),
    "loss": (0.0, 80.0),
}
b0 = st.session_state.bases[phe_now]
DEF = {"th": b0[0], "tc": b0[1], "vl": vl0, "vw": vw0, "loss": 0.0}
for k, (lo, hi) in RNG.items():
    st.session_state[k] = float(np.clip(st.session_state.get(k, DEF[k]), lo, hi))

with st.sidebar:
    st.header("⚙️ Simulator Controls")
    phe = st.radio("เครื่องที่ต้องการจำลอง", PHES, key="phe", horizontal=True, on_change=cb_reset)
    st.selectbox("Fault Scenario (preset)", list(SCEN), key="scen", on_change=cb_scen,
                 help="เลือกแล้วระบบจะตั้งค่าทุกตัวให้ จากนั้นยังปรับละเอียดเองต่อได้")
    times = D[D.phe == phe].dropna(subset=["th_i"]).Time.dt.strftime("%d %b %H:%M").tolist()
    st.selectbox("Replay จากข้อมูลจริง (เลือกเวลา)", ["— ไม่ใช้ —"] + times, key="replay", on_change=cb_replay,
                 help="ดึงอุณหภูมิขาเข้าจริง ณ เวลานั้นมาใส่ในตัวจำลอง")
    st.button("↺ Reset เป็นค่าปกติ", on_click=cb_reset, use_container_width=True)

    st.markdown("**อุณหภูมิขาเข้า**")
    st.slider("Lye inlet (°C)", *RNG["th"], step=0.1, key="th",
              help=f"ข้อมูลจริง {rg['th_i'][0]:.1f}–{rg['th_i'][1]:.1f} °C")
    st.slider("Cooling water inlet (°C)", *RNG["tc"], step=0.1, key="tc",
              help=f"ข้อมูลจริง {rg['tc_i'][0]:.1f}–{rg['tc_i'][1]:.1f} °C")
    st.markdown("**อัตราการไหล**")
    st.slider("Lye flow (m³/hr)", *RNG["vl"], step=0.1, key="vl")
    st.slider("Cooling water flow (m³/hr)", *RNG["vw"], step=0.1, key="vw")
    st.markdown("**สภาพเครื่อง**")
    st.slider("UA degradation / fouling (%)", *RNG["loss"], step=1.0, key="loss",
              help="0% = เครื่องสะอาดเท่าค่า UA เฉลี่ยจากข้อมูลจริง")

th, tc, vl, vw, loss = (st.session_state[k] for k in ("th", "tc", "vl", "vw", "loss"))

# ============================================================
# คำนวณ Simulation + AI baseline
# ============================================================
m = M[phe]
ua_sim = m["ua_med"] * (1 - loss / 100)
th_o, tc_o, q, eps, ntu = predict(th, tc, ua_sim, vl, vw)
c = calc(th, th_o, tc, tc_o, vl, vw)
b_tho, b_tco, b_q, b_eps, _ = predict(th, tc, m["ua_med"], vl0, vw0)  # เครื่องปกติ สภาวะเดียวกัน

exp_tho = float(m["rf"].predict(pd.DataFrame([[th, tc]], columns=["th_i", "tc_i"]).values)[0])
z = (th_o - exp_tho) / m["sigma"]
feat = pd.DataFrame([[th, tc, th_o, float(c["eps"])]], columns=FEATS_IF)
iso_score = float(m["iso"].decision_function(feat)[0])
iso_flag = bool(m["iso"].predict(feat)[0] == -1)
ua_ratio = float(c["UA"]) / m["ua_med"]
inside = all(m["rng"][k][0] - 0.5 <= v <= m["rng"][k][1] + 0.5 for k, v in (("th_i", th), ("tc_i", tc)))

tab1, tab2, tab3 = st.tabs(["1️⃣ Plant Data & AI Baseline", "2️⃣ Interactive Simulator", "3️⃣ AI Diagnostics"])

# ------------------------------------------------------------
with tab1:
    st.subheader("ข้อมูลจริงจากโรงงานและโมเดลที่เทรนจากข้อมูลนี้")
    a, b, c3, d4 = st.columns(4)
    a.metric("จำนวนข้อมูลจริง", f"{len(raw)} แถว")
    b.metric("ช่วงเวลา", f"{raw.Time.min():%d %b} – {raw.Time.max():%d %b %Y}")
    c3.metric("ความดันผิดปกติ (ตัดออก)", f"{int(D[D.phe == 'PHE#1'].inv.sum() + D[D.phe == 'PHE#2'].inv.sum())} จุด")
    d4.metric("Anomaly ที่ AI พบ", f"{int(sum(M[p]['scored'].anom.sum() for p in PHES))} จุด")
    st.warning("อัตราการไหลเป็นค่าสมมติ (Lye 7, Water 10 m³/hr) → Q, UA, ε เป็นค่าประมาณ "
               "แต่ค่า ΔT, อุณหภูมิ และแนวโน้มเชิงเปรียบเทียบยังใช้ได้")

    st.markdown("#### UA และ ΔP ตามเวลา (จุด X = ผิดปกติตาม Isolation Forest)")
    g1, g2 = st.columns(2)
    f1, f2 = go.Figure(), go.Figure()
    for p in PHES:
        s = M[p]["scored"]
        f1.add_scatter(x=s.Time, y=s.UA, mode="lines", name=f"{p} UA", opacity=0.35)
        f1.add_scatter(x=s.Time, y=s.UA.rolling(12, center=True, min_periods=4).median(), name=f"{p} เฉลี่ยเคลื่อนที่ 1 วัน")
        a_ = s[s.anom]
        f1.add_scatter(x=a_.Time, y=a_.UA, mode="markers", marker_symbol="x", marker_size=9, name=f"{p} anomaly")
        f2.add_scatter(x=s.Time, y=s.dP_l.rolling(12, center=True, min_periods=4).mean(), name=f"{p} ΔP lye")
        f2.add_scatter(x=s.Time, y=s.dP_w.rolling(12, center=True, min_periods=4).mean(), name=f"{p} ΔP water", line_dash="dot")
    f1.update_layout(yaxis_title="UA (W/K)", height=360, legend=dict(orientation="h", y=-0.25))
    f2.update_layout(yaxis_title="ΔP (kg/cm²)", height=360, legend=dict(orientation="h", y=-0.25))
    g1.plotly_chart(f1, use_container_width=True)
    g2.plotly_chart(f2, use_container_width=True)

    st.markdown("#### แนวโน้ม Fouling (ถดถอยเชิงเส้นของ UA ตามเวลา)")
    rows = []
    for p in PHES:
        s = M[p]["scored"]
        lr = linregress(s.day, s.UA)
        pct = lr.slope / s.UA.median() * 100
        rows.append({"เครื่อง": p, "UA เฉลี่ย (W/K)": round(s.UA.median()), "ε เฉลี่ย": round(s.eps.mean(), 3),
                     "แนวโน้ม UA (%/วัน)": round(pct, 2), "p-value": round(lr.pvalue, 3),
                     "สรุป": "ลดลงอย่างมีนัยสำคัญ" if (lr.pvalue < 0.05 and pct < 0) else "ยังไม่พบแนวโน้ม fouling"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### ความแม่นยำของโมเดล (Time-series cross-validation)")
    cA, cB = st.columns(2)
    for col, p in zip((cA, cB), PHES):
        col.markdown(f"**{p}**")
        col.dataframe(M[p]["metrics"], use_container_width=True)
    st.caption("เทียบ AI (Random Forest) กับ Physics และ Naive โดยทดสอบกับช่วงเวลาที่โมเดลไม่เคยเห็น "
               "ข้อมูลอ่านมือละเอียด 1 °C จึงมี noise ระดับ ~0.5–1 °C")

    with st.expander("ดูข้อมูลดิบและตารางที่คำนวณแล้ว"):
        st.dataframe(D.drop(columns=["inv"]).round(3), use_container_width=True)
        st.download_button("ดาวน์โหลดข้อมูลที่คำนวณแล้ว (CSV)", D.to_csv(index=False).encode("utf-8-sig"), "phe_derived.csv")

    an = pd.concat([M[p]["scored"].assign(phe=p) for p in PHES]).query("anom").sort_values("score").head(10)
    with st.expander("เวลาที่ AI มองว่าผิดปกติที่สุด (Top 10)"):
        st.dataframe(an[["Time", "phe", "th_i", "th_o", "tc_i", "tc_o", "eps", "UA", "score"]].round(3),
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------
with tab2:
    st.subheader(f"ผลการจำลอง — {phe}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Heat Duty", f"{q:.1f} kW", f"{q - b_q:+.1f} kW เทียบปกติ")
    k2.metric("Lye outlet", f"{th_o:.1f} °C", f"{th_o - b_tho:+.1f} °C เทียบปกติ", delta_color="inverse")
    k3.metric("Water outlet", f"{tc_o:.1f} °C", f"{tc_o - b_tco:+.1f} °C เทียบปกติ", delta_color="off")
    k4.metric("Effectiveness ε", f"{eps:.3f}", f"{eps - b_eps:+.3f} เทียบปกติ")
    k5.metric("UA จริงที่ใช้", f"{ua_sim:,.0f} W/K", f"{-loss:.0f}% จากค่าปกติ" if loss else "ปกติ", delta_color="inverse" if loss else "off")
    st.caption(f"\"เทียบปกติ\" = เครื่องสะอาด (UA เฉลี่ยจากข้อมูลจริง {m['ua_med']:,.0f} W/K) ที่อุณหภูมิขาเข้าเดียวกันและอัตราไหลฐาน")
    if not inside:
        st.warning("อุณหภูมิขาเข้าอยู่นอกช่วงข้อมูลที่ AI เคยเห็น — ผลของโมเดล AI ในแท็บ 3 อาจไม่น่าเชื่อถือ")

    g1, g2 = st.columns(2)
    fb = go.Figure()
    fb.add_bar(name="ปกติ (baseline)", x=["Lye outlet", "Water outlet"], y=[b_tho, b_tco])
    fb.add_bar(name="จำลอง", x=["Lye outlet", "Water outlet"], y=[th_o, tc_o])
    fb.update_layout(barmode="group", yaxis_title="°C", title="อุณหภูมิขาออก: ปกติ vs จำลอง", height=360)
    g1.plotly_chart(fb, use_container_width=True)

    s = m["scored"]
    fs = go.Figure()
    fs.add_scatter(x=s.th_i, y=s.th_o, mode="markers", name="ข้อมูลจริง", opacity=0.5)
    fs.add_scatter(x=[th], y=[th_o], mode="markers", name="จำลอง", marker=dict(symbol="star", size=16, color="red"))
    fs.update_layout(xaxis_title="Lye inlet (°C)", yaxis_title="Lye outlet (°C)", title="จุดจำลองเทียบกับข้อมูลจริง", height=360)
    g2.plotly_chart(fs, use_container_width=True)

    xs = np.arange(0, 81)
    ys = [predict(th, tc, m["ua_med"] * (1 - x / 100), vl, vw)[0] for x in xs]
    fz = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="Lye outlet"))
    fz.add_scatter(x=[loss], y=[th_o], mode="markers", marker=dict(size=12, color="red"), name="ตอนนี้")
    fz.update_layout(xaxis_title="UA degradation (%)", yaxis_title="Lye outlet (°C)",
                     title="ความไว: Fouling ทำให้ Lye ออกร้อนขึ้นเท่าไร (ที่ค่าปัจจุบัน)", height=320)
    st.plotly_chart(fz, use_container_width=True)

# ------------------------------------------------------------
with tab3:
    st.subheader("AI Diagnostics & Decision Support")
    health = float(np.clip(ua_ratio * 100, 0, 100))
    critical = ua_ratio < 0.6 or abs(z) > 4 or (iso_flag and abs(z) > 3)
    warning = ua_ratio < 0.85 or abs(z) > 2 or iso_flag
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Health Index (UA เทียบปกติ)", f"{health:.0f} %")
    h2.metric("Lye outlet เทียบที่ AI คาด", f"{th_o - exp_tho:+.2f} °C", f"{z:+.1f} σ", delta_color="off")
    h3.metric("Isolation Forest score", f"{iso_score:+.3f}", "ผิดปกติ" if iso_flag else "ปกติ", delta_color="inverse" if iso_flag else "off")
    h4.metric("ช่วงข้อมูลที่เคยเรียนรู้", "อยู่ในช่วง" if inside else "นอกช่วง", delta_color="off")

    if critical:
        st.error("### 🚨 CRITICAL: ตรวจพบความผิดปกติระดับวิกฤต")
    elif warning:
        st.warning("### ⚠️ WARNING: พฤติกรรมเบี่ยงเบนจากค่าปกติ")
    else:
        st.success("### ✅ NORMAL: สอดคล้องกับพฤติกรรมปกติที่เรียนรู้จากข้อมูลจริง")

    why = []
    if loss >= 15 or ua_ratio < 0.85:
        why.append(f"UA ลดลงเหลือ ~{ua_ratio * 100:.0f}% ของค่าปกติ → สอดคล้องกับ **fouling** ฝั่งใดฝั่งหนึ่ง (ตรวจ ΔP ทั้ง 2 ฝั่งควบคู่)")
    if vl / vl0 < 0.8:
        why.append(f"Lye flow ลดเหลือ {vl / vl0 * 100:.0f}% ของฐาน → ตรวจปั๊ม/วาล์ว/ไส้กรองฝั่ง lye")
    if vw / vw0 < 0.8:
        why.append(f"น้ำหล่อเย็นลดเหลือ {vw / vw0 * 100:.0f}% ของฐาน → ตรวจปั๊ม/วาล์วน้ำ")
    if tc - b0[1] > 2:
        why.append(f"น้ำเข้าร้อนขึ้น {tc - b0[1]:+.1f} °C → ตรวจ cooling tower/อุณหภูมิน้ำป้อน")
    if th - b0[0] > 3:
        why.append(f"Lye เข้าร้อนขึ้น {th - b0[0]:+.1f} °C → ตรวจกระบวนการต้นทาง")
    if not inside:
        why.append("อุณหภูมิขาเข้านอกช่วงข้อมูลจริง → ควรเก็บข้อมูลเพิ่มก่อนเชื่อผล AI")
    if why:
        st.markdown("**สาเหตุที่เป็นไปได้ / คำแนะนำ**")
        for w in why:
            st.markdown(f"- {w}")
    elif not warning:
        st.markdown("ไม่พบสิ่งที่ต้องดำเนินการ ณ สภาวะที่จำลอง")

    st.divider()
    st.markdown("**วิธีที่ระบบตัดสิน:** (1) Health Index = UA ที่ได้จากการจำลอง ÷ UA ปกติจากข้อมูลจริง, "
                "(2) Random Forest ที่เทรนจากข้อมูลจริงทำนาย Lye outlet ที่ \"ควรเป็น\" แล้ววัดส่วนต่างเป็นหน่วย σ ของ error ที่ทดสอบข้ามเวลา, "
                "(3) Isolation Forest ตรวจว่าสภาวะรวม (T, ε) อยู่นอกรูปแบบที่เคยเห็นหรือไม่")
    st.caption("หมายเหตุ: ข้อมูลจริง 14 วันเป็นสภาพเครื่องที่ยังปกติ จึงเรียนรู้ได้เฉพาะ 'ลักษณะปกติ' — "
               "สถานการณ์ fouling/flow drop มาจากแบบจำลองฟิสิกส์ ไม่ใช่ข้อมูลความเสียหายจริง")
