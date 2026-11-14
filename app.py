import io
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import linregress
import warnings
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
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


def _ml(name):
    if name == "RandomForest":
        return RandomForestRegressor(200, min_samples_leaf=5, random_state=0)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
    if name == "Ridge":
        return Ridge(1.0)
    return GaussianProcessRegressor(ConstantKernel(1.0) * RBF([5.0, 5.0]) + WhiteKernel(0.5), normalize_y=True, random_state=0)


@st.cache_data(show_spinner="กำลังเทรนโมเดลจากข้อมูลจริง...")
def train(D: pd.DataFrame, phe: str, vl: float, vw: float) -> dict:
    s = D[D.phe == phe].dropna(subset=FEATS_IF + ["Q"])
    s = s[(s.Q > 0) & (s.UA > 0)].reset_index(drop=True)
    X, y = s[["th_i", "tc_i"]].values, s.th_o.values

    # ประเมินแบบ time-series CV (ทดสอบกับช่วงเวลาที่โมเดลไม่เคยเห็น)
    ML = ["RandomForest", "GradientBoosting", "Ridge", "GaussianProcess"]
    P = {k: np.full(len(s), np.nan) for k in ML + ["Physics ε-NTU", "Hybrid (Physics+RF)", "Naive (ΔT เฉลี่ย)"]}
    for tr, te in TimeSeriesSplit(5).split(X):
        ua0 = np.median(s.UA.values[tr])
        ph_tr = predict(s.th_i.values[tr], s.tc_i.values[tr], ua0, vl, vw)[0]
        ph_te = predict(s.th_i.values[te], s.tc_i.values[te], ua0, vl, vw)[0]
        for k in ML:
            P[k][te] = _ml(k).fit(X[tr], y[tr]).predict(X[te])
        P["Physics ε-NTU"][te] = ph_te
        P["Hybrid (Physics+RF)"][te] = ph_te + _ml("RandomForest").fit(X[tr], y[tr] - ph_tr).predict(X[te])
        P["Naive (ΔT เฉลี่ย)"][te] = s.th_i.values[te] - np.mean(s.th_i.values[tr] - y[tr])
    mask = ~np.isnan(P["Naive (ΔT เฉลี่ย)"])
    rows = {}
    for k, p in P.items():
        e = p[mask] - y[mask]
        rows[k] = {"MAE (°C)": np.mean(np.abs(e)), "RMSE (°C)": np.sqrt(np.mean(e ** 2)), "R²": r2_score(y[mask], p[mask])}
    metrics = pd.DataFrame(rows).T.round(3)
    cv = pd.DataFrame({"Time": s.Time, "actual": y, **P})[mask].reset_index(drop=True)
    sigma = max(float(np.std(P["RandomForest"][mask] - y[mask])), 0.3)

    rf = _ml("RandomForest").fit(X, y)
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, contamination=0.05, random_state=0))
    iso.fit(s[FEATS_IF])
    s["score"] = iso.decision_function(s[FEATS_IF])
    s["anom"] = iso.predict(s[FEATS_IF]) == -1
    return dict(
        rf=rf, iso=iso, sigma=sigma, metrics=metrics, cv=cv, scored=s,
        ua_med=float(s.UA.median()), ua_lo=float(s.UA.quantile(.05)), ua_hi=float(s.UA.quantile(.95)),
        rng={c: (float(s[c].min()), float(s[c].max())) for c in ("th_i", "tc_i", "th_o")},
    )


@st.cache_data(show_spinner=False)
def ua_band(s: pd.DataFrame, vl: float, vw: float, n: int = 300):
    """Monte Carlo: ใส่ความคลาดเคลื่อนการอ่านมือ ±0.5 °C แล้วดูช่วง 5–95% ของ UA"""
    r = np.random.default_rng(0)
    sims = []
    for _ in range(n):
        e = {c: s[c].values + r.uniform(-0.5, 0.5, len(s)) for c in ("th_i", "th_o", "tc_i", "tc_o")}
        sims.append(calc(e["th_i"], e["th_o"], e["tc_i"], e["tc_o"], vl, vw)["UA"])
    sims = np.array(sims, float)
    return np.nanpercentile(sims, 5, axis=0), np.nanpercentile(sims, 95, axis=0)


@st.cache_data(show_spinner=False)
def load_log(blob: bytes) -> pd.DataFrame:
    try:
        x = pd.read_excel(io.BytesIO(blob), sheet_name="Summary Statistics", header=None)
        i = x.index[x[0].astype(str).str.contains("DATA QUALITY LOG")][0]
        rows = []
        for j in range(i + 1, len(x)):
            if pd.isna(x.iloc[j, 0]):
                break
            rows.append({"สถานะ": x.iloc[j, 0], "เวลา": x.iloc[j, 1],
                         "รายละเอียด": next((v for v in x.iloc[j, 2:] if pd.notna(v)), "")})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner="กำลังคำนวณ sensitivity...")
def sens_table(raw: pd.DataFrame, vl: float, vw: float) -> pd.DataFrame:
    rows = []
    for fl in (0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3):
        for fw in (0.7, 1.0, 1.3):
            Dx = derive(raw, vl * fl, vw * fw)
            r = {"Lye flow ×": fl, "Water flow ×": fw}
            for p in PHES:
                s = Dx[Dx.phe == p].dropna(subset=["UA", "eps", "day"])
                s = s[s.UA > 0]
                lr = linregress(s.day, s.UA)
                r[f"UA {p}"] = round(s.UA.median())
                r[f"ε {p}"] = round(s.eps.mean(), 3)
                r[f"แนวโน้ม UA {p} (%/วัน)"] = round(lr.slope / s.UA.median() * 100, 2)
                r[f"p {p}"] = round(lr.pvalue, 3)
            rows.append(r)
    return pd.DataFrame(rows)


def inject(s, sev, vl, vw, ua_med, rng):
    """Fault injection: ใช้อุณหภูมิขาเข้าจริง แต่ทำให้ UA ลดลง sev (สัดส่วน) แล้วสร้างค่าที่ 'วัดได้' (ปัดเป็น 1 °C + noise)"""
    tho, tco = predict(s.th_i.values, s.tc_i.values, ua_med * (1 - sev), vl, vw)[:2]
    tho = np.round(tho + rng.normal(0, .3, len(s)))
    tco = np.round(tco + rng.normal(0, .3, len(s)))
    c = calc(s.th_i.values, tho, s.tc_i.values, tco, vl, vw)
    return pd.DataFrame({"th_i": s.th_i.values, "tc_i": s.tc_i.values, "th_o": tho, "tc_o": tco, "eps": c["eps"], "UA": c["UA"]})


def detect(df, m):
    z = (df.th_o.values - m["rf"].predict(df[["th_i", "tc_i"]].values)) / m["sigma"]
    sd = m["scored"].UA.std()
    hi = pd.Series(df.UA.values).rolling(12, min_periods=6).median().values / m["ua_med"]
    return {
        "RF residual > 2σ": z > 2,
        "Isolation Forest": m["iso"].predict(df[FEATS_IF]) == -1,
        "Control chart UA < μ−2σ": df.UA.values < m["ua_med"] - 2 * sd,
        "Health Index (12 จุด) < 85%": hi < 0.85,
    }


def cb_use_est():
    v = st.session_state.get("_vw_est")
    if v and v == v:
        v = float(np.clip(v, 0.5, 80.0))
        st.session_state.vw0, st.session_state.vw = v, v


CLASSES = ["Normal", "Fouling", "Lye flow drop", "Water flow drop", "Warm cooling water", "Hot lye surge"]
CF = ["dth", "dtc", "resid_l", "resid_w"]


def predict_v(th_i, tc_i, ua, vl, vw):
    """ε-NTU แบบ vectorized (รองรับ flow/UA เป็นอาร์เรย์) — ใช้สร้างข้อมูลสังเคราะห์"""
    cl, cw = caps(vl, vw)
    cmin, cmax = np.minimum(cl, cw), np.maximum(cl, cw)
    cr, ntu = cmin / cmax, ua / cmin
    e = np.exp(-ntu * (1 - cr))
    with np.errstate(all="ignore"):
        eps = np.where(np.abs(1 - cr) < 1e-9, ntu / (1 + ntu), (1 - e) / (1 - cr * e))
    q = eps * cmin * (th_i - tc_i)
    return th_i - q / cl, tc_i + q / cw


def decide(clf, X, thr=0.6):
    """จัดเป็นความผิดปกติเฉพาะเมื่อมั่นใจ ≥ thr มิฉะนั้นถือเป็น Normal (ลด false alarm)"""
    pr = clf.predict_proba(X)
    lab = clf.classes_[pr.argmax(1)].astype(object)
    lab[(lab != "Normal") & (pr.max(1) < thr)] = "Normal"
    return lab, pr


def synth_faults(s, ua_med, vl, vw, base, sig_l, sig_w, n_per, seed=0):
    """สร้างข้อมูลสังเคราะห์ 6 คลาส: ใช้อุณหภูมิขาเข้าจริงสุ่มซ้ำ + ฟิสิกส์ + noise เท่าที่เห็นในข้อมูลจริง"""
    r = np.random.default_rng(seed)
    out = []
    for cname in CLASSES:
        idx = r.integers(0, len(s), n_per)
        thi, tci = s.th_i.values[idx].astype(float), s.tc_i.values[idx].astype(float)
        ua, fl, fw = np.full(n_per, ua_med), np.full(n_per, vl), np.full(n_per, vw)
        sev = np.full(n_per, np.nan)
        if cname == "Fouling":
            sev = r.uniform(.15, .6, n_per)
            ua = ua_med * (1 - sev)
        elif cname == "Lye flow drop":
            fl = vl * r.uniform(.4, .8, n_per)
        elif cname == "Water flow drop":
            fw = vw * r.uniform(.4, .8, n_per)
        elif cname == "Warm cooling water":
            tci = tci + r.uniform(3, 8, n_per)
        elif cname == "Hot lye surge":
            thi = thi + r.uniform(3, 8, n_per)
        tho_f, tco_f = predict_v(thi, tci, ua, fl, fw)
        tho_h, tco_h = predict_v(thi, tci, ua_med, vl, vw)
        out.append(pd.DataFrame({
            "cls": cname, "sev": sev, "dth": thi - base[0], "dtc": tci - base[1],
            "resid_l": tho_f - tho_h + r.normal(0, sig_l, n_per),
            "resid_w": tco_f - tco_h + r.normal(0, sig_w, n_per)}))
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner="กำลังเทรน Fault Classifier...")
def build_clf(s: pd.DataFrame, ua_med: float, vl: float, vw: float) -> dict:
    X = s[["th_i", "tc_i"]].values
    kf = KFold(5)
    rl = s.th_o.values - cross_val_predict(_ml("RandomForest"), X, s.th_o.values, cv=kf)
    rw = s.tc_o.values - cross_val_predict(_ml("RandomForest"), X, s.tc_o.values, cv=kf)
    sig_l, sig_w = max(float(rl.std()), .3), max(float(rw.std()), .3)
    base = (float(s.th_i.median()), float(s.tc_i.median()))
    syn = synth_faults(s, ua_med, vl, vw, base, sig_l, sig_w, 600)
    Xtr, Xte, ytr, yte, _, ste = train_test_split(syn[CF], syn.cls, syn.sev, test_size=.3, stratify=syn.cls, random_state=0)
    clf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0).fit(Xtr, ytr)
    pred = decide(clf, Xte)[0]
    cm = pd.DataFrame(confusion_matrix(yte, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
    rec = []
    for lo, hi in [(.15, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .61)]:
        mk = ((yte == "Fouling") & (ste >= lo) & (ste < hi)).values
        rec.append({"UA ลดลง (%)": f"{lo * 100:.0f}–{min(hi, .6) * 100:.0f}", "Recall ของ Fouling (%)": round(100 * float(np.mean(pred[mk] == "Fouling")), 1)})
    real = pd.DataFrame({"dth": s.th_i.values - base[0], "dtc": s.tc_i.values - base[1], "resid_l": rl, "resid_w": rw})
    rp = decide(clf, real[CF])[0]
    dist = (pd.Series(rp).value_counts(normalize=True).reindex(CLASSES).fillna(0) * 100).round(1)
    return dict(clf=clf, cm=cm, acc=float(accuracy_score(yte, pred)), rec=pd.DataFrame(rec), dist=dist,
                base=base, sig=(sig_l, sig_w), real_pred=rp, imp=dict(zip(CF, clf.feature_importances_.round(3))))


# ============================================================
# Sidebar: แหล่งข้อมูล
# ============================================================
st.title("🌡️ AI-Based Heat Exchanger Digital Twin")
st.caption("Digitalization in Chemical Engineering — Caustic Absorption Plant (PHE#1 / PHE#2, Lye 43% NaOH ↔ Cooling Water)")

with st.sidebar:
    st.header("📁 แหล่งข้อมูล")
    ups = st.file_uploader("อัปโหลดไฟล์ PHE_Performance_Analysis (.xlsx) — เลือกได้หลายไฟล์ ระบบจะรวมให้", type=["xlsx"],
                           accept_multiple_files=True)  # ต้องอัปโหลดก่อนจึงจะแสดงผล
    with st.expander("⚠️ อัตราการไหลฐาน (ค่าสมมติ)"):
        st.number_input("Lye flow (m³/hr)", 0.5, 50.0, 7.0, 0.5, key="vl0")
        st.number_input("Cooling water flow (m³/hr)", 0.5, 80.0, 10.0, 0.5, key="vw0")
        st.caption("ยังไม่มีค่าจริงจาก pump nameplate — เมื่อได้ค่าจริงให้แก้ตรงนี้ โมเดลทั้งหมดจะคำนวณใหม่")
    with st.expander("ข้อมูล datasheet (ถ้ามี) → คำนวณ Rf"):
        st.number_input("พื้นที่แลกเปลี่ยนความร้อน A (m²) — 0 = ไม่ทราบ", 0.0, 10000.0, 0.0, 1.0, key="A_hx")
        st.number_input("U_clean (W/m²K) — 0 = ไม่ทราบ", 0.0, 10000.0, 0.0, 10.0, key="U_clean")

if not ups:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูล Plant Data (.xlsx) ที่แถบด้านซ้ายก่อน ระบบจึงจะวิเคราะห์และเปิดตัวจำลองครับ")
    st.stop()
try:
    raw = (pd.concat([load_plant(u.getvalue()) for u in ups]).drop_duplicates("Time")
           .sort_values("Time").reset_index(drop=True))
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

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Overview", "1️⃣ Plant Data & AI Baseline", "2️⃣ Interactive Simulator", "3️⃣ AI Diagnostics",
                                              "4️⃣ Fouling Lab", "5️⃣ Flow Sensitivity", "6️⃣ Model & Limitations", "7️⃣ Fault Classifier"])

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
    st.caption("เทียบ AI 4 แบบ, Physics, Hybrid (Physics+RF) และ Naive โดยทดสอบกับช่วงเวลาที่โมเดลไม่เคยเห็น "
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


# ============================================================
# เพิ่มเติมในแท็บ 1: Validation, ความไม่แน่นอนของ UA, Data cleaning
# ============================================================
with tab1:
    st.divider()
    st.subheader("Validation: โมเดลตรงกับของจริงแค่ไหน")
    v1, v2 = st.columns([1, 2])
    vphe = v1.radio("เครื่อง", PHES, key="val_phe", horizontal=True)
    vmod = v2.selectbox("โมเดลที่ต้องการดู", list(M[vphe]["metrics"].index), index=0, key="val_model")
    cv = M[vphe]["cv"]
    fv = go.Figure()
    fv.add_scatter(x=cv.actual, y=cv[vmod], mode="markers", name="CV prediction", opacity=0.6)
    lo_, hi_ = float(cv.actual.min()) - 1, float(cv.actual.max()) + 1
    fv.add_scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", name="y = x", line_dash="dash")
    fv.update_layout(xaxis_title="Lye outlet จริง (°C)", yaxis_title="Lye outlet ที่ทำนาย (°C)", height=380,
                     title=f"{vphe}: {vmod} (ทดสอบข้ามเวลา)")
    st.plotly_chart(fv, use_container_width=True)
    sv = M[vphe]["scored"]
    tco_p = predict(sv.th_i.values, sv.tc_i.values, M[vphe]["ua_med"], vl0, vw0)[1]
    st.caption(f"MAE ของ Water outlet จากแบบจำลองฟิสิกส์ = {np.mean(np.abs(tco_p - sv.tc_o.values)):.2f} °C "
               "(สูงกว่าฝั่ง lye เพราะ flow น้ำเป็นค่าสมมติ — ดูแท็บ Flow Sensitivity)")

    st.subheader("ความไม่แน่นอนของ UA (Monte Carlo ±0.5 °C)")
    lo_b, hi_b = ua_band(sv, vl0, vw0)
    fu = go.Figure()
    fu.add_scatter(x=sv.Time, y=hi_b, mode="lines", line_width=0, showlegend=False)
    fu.add_scatter(x=sv.Time, y=lo_b, mode="lines", line_width=0, fill="tonexty", name="ช่วง 5–95%")
    fu.add_scatter(x=sv.Time, y=sv.UA, mode="lines+markers", name="UA จากข้อมูล", marker_size=4)
    fu.update_layout(yaxis_title="UA (W/K)", height=360, title=f"{vphe}: UA พร้อมช่วงความไม่แน่นอน")
    st.plotly_chart(fu, use_container_width=True)
    rel = np.nanmedian((hi_b - lo_b) / 2 / sv.UA.values) * 100
    st.info(f"ความคลาดเคลื่อนการอ่านมือ ±0.5 °C ทำให้ UA ของแต่ละจุดไม่แน่นอนประมาณ ±{rel:.0f}% "
            "ดังนั้นควรตัดสินแนวโน้มจากค่าเฉลี่ยเคลื่อนที่ ไม่ใช่จุดเดียว")

    st.subheader("ตารางสรุปการทำความสะอาดข้อมูล")
    log = load_log(ups[0].getvalue())
    if len(log):
        st.dataframe(log, hide_index=True, use_container_width=True)
    eff = []
    for p_ in (1, 2):
        before = (raw[f"P{p_}_li"] - raw[f"P{p_}_lo"]).median()
        after = D[D.phe == f"PHE#{p_}"].dP_l.median()
        eff.append({"เครื่อง": f"PHE#{p_}", "ความดันผิดปกติที่ตัดออก (จุด)": int(D[D.phe == f"PHE#{p_}"].inv.sum()),
                    "median ΔP lye ก่อนตัด": round(before, 3), "หลังตัด": round(after, 3)})
    st.dataframe(pd.DataFrame(eff), hide_index=True, use_container_width=True)

# ============================================================
# เพิ่มเติมในแท็บ 2: ΔP
# ============================================================
with tab2:
    st.markdown("#### ความดันตกคร่อม ΔP (แบบจำลองสมมติ)")
    kp = st.number_input("ความไวของ ΔP ต่อ fouling (kp)", 0.0, 5.0, 1.0, 0.1, key="kp",
                         help="ΔP ถูกคูณด้วย (1 + kp × fouling%) — ปรับได้เมื่อมีข้อมูลช่วงล้างเครื่องจริง")
    sp = m["scored"]
    dpl0, dpw0 = float(sp.dP_l.median()), float(sp.dP_w.median())
    fr_ = 1 + kp * loss / 100
    dpl, dpw = dpl0 * (vl / vl0) ** 1.8 * fr_, dpw0 * (vw / vw0) ** 1.8 * fr_
    p1, p2 = st.columns(2)
    p1.metric("ΔP ฝั่ง lye", f"{dpl:.2f} kg/cm²", f"{dpl - dpl0:+.2f} เทียบปกติ", delta_color="inverse")
    p2.metric("ΔP ฝั่งน้ำ", f"{dpw:.2f} kg/cm²", f"{dpw - dpw0:+.2f} เทียบปกติ", delta_color="inverse")
    st.caption("สมมติ ΔP ∝ flow^1.8 (ไหลปั่นป่วน) และเพิ่มตาม fouling; ค่าปกติใช้ median จากข้อมูลจริง")

# ============================================================
# แท็บ 4: Fouling Lab
# ============================================================
with tab4:
    st.subheader("Fouling Lab: ข้อมูลสังเคราะห์, Fault Injection, RUL, เศรษฐศาสตร์")
    st.warning("ส่วนนี้ใช้ข้อมูลสังเคราะห์จากแบบจำลองฟิสิกส์ + อุณหภูมิขาเข้าจริงที่สุ่มซ้ำ — "
               "ไม่ใช่ข้อมูล fouling จริงของโรงงาน ต้องระบุเช่นนี้ในรายงาน")
    c1, c2, c3, c4 = st.columns(4)
    lab_phe = c1.radio("เครื่อง", PHES, key="lab_phe", horizontal=True)
    days = c2.slider("ระยะเวลา (วัน)", 20, 120, 60, key="lab_days")
    linf = c3.slider("UA ลดลงสูงสุด (%)", 10, 70, 40, key="lab_linf")
    tau = c4.slider("ค่าคงที่เวลา τ (วัน)", 5, 60, 25, key="lab_tau")
    ml, sl = M[lab_phe], M[lab_phe]["scored"]
    r_ = np.random.default_rng(1)
    n = days * 12
    idx = r_.integers(0, len(sl), n)
    t = np.arange(n) / 12
    loss_t = linf / 100 * (1 - np.exp(-t / tau))
    thi, tci = sl.th_i.values[idx], sl.tc_i.values[idx]
    tho, tco = predict(thi, tci, ml["ua_med"] * (1 - loss_t), vl0, vw0)[:2]
    tho = np.round(tho + r_.normal(0, .3, n))
    tco = np.round(tco + r_.normal(0, .3, n))
    cc = calc(thi, tho, tci, tco, vl0, vw0)
    syn = pd.DataFrame({"t": t, "UA": cc["UA"], "th_o": tho})
    syn["HI"] = syn.UA.rolling(24, min_periods=12).median() / ml["ua_med"]
    syn["HI_true"] = 1 - loss_t
    kp_ = st.session_state.get("kp", 1.0)
    dpl_t = float(sl.dP_l.median()) * (1 + kp_ * loss_t)

    def first_day(cond):
        cond = np.asarray(cond)
        return float(t[np.argmax(cond)]) if cond.any() else None

    w_det, w_true = first_day(syn.HI.values < 0.85), first_day(syn.HI_true.values < 0.85)
    c_det, c_true = first_day(syn.HI.values < 0.70), first_day(syn.HI_true.values < 0.70)
    fmt = lambda v: "ไม่ถึงเกณฑ์" if v is None else f"วันที่ {v:.1f}"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("เตือน (HI < 85%) — AI ตรวจพบ", fmt(w_det))
    k2.metric("เตือน — จริง", fmt(w_true))
    k3.metric("ควรล้าง (HI < 70%) — AI ตรวจพบ", fmt(c_det))
    k4.metric("ควรล้าง — จริง", fmt(c_true))
    tail = syn.dropna(subset=["HI"]).tail(84)
    if len(tail) > 10:
        sl_, ic_ = np.polyfit(tail.t, tail.HI, 1)
        rul = (0.70 - ic_) / sl_ if sl_ < 0 else None
        st.caption("RUL แบบเส้นตรงจาก 7 วันล่าสุด: " + (f"HI ถึง 70% ราววันที่ {rul:.0f} (เหลืออีก ~{max(rul - days, 0):.0f} วัน)"
                   if rul else "ยังไม่มีแนวโน้มลดลง") + " — เส้นตรงอาจประเมินสูงเกินไปเมื่อ fouling เข้าใกล้ค่าอิ่มตัว")

    g1, g2 = st.columns(2)
    f1 = go.Figure()
    f1.add_scatter(x=syn.t, y=syn.HI, name="HI ที่ AI วัดได้ (median 2 วัน)")
    f1.add_scatter(x=syn.t, y=syn.HI_true, name="HI จริงในแบบจำลอง", line_dash="dash")
    f1.add_hline(y=0.85, line_dash="dot", annotation_text="เตือน")
    f1.add_hline(y=0.70, line_dash="dot", annotation_text="ล้าง")
    f1.update_layout(xaxis_title="วัน", yaxis_title="Health Index", height=360)
    g1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure()
    f2.add_scatter(x=t, y=pd.Series(tho).rolling(24, min_periods=6).mean(), name="Lye outlet (°C)")
    f2.add_scatter(x=t, y=dpl_t, name="ΔP lye (kg/cm²)", yaxis="y2")
    f2.update_layout(xaxis_title="วัน", yaxis_title="Lye outlet (°C)", height=360,
                     yaxis2=dict(title="ΔP (kg/cm²)", overlaying="y", side="right"))
    g2.plotly_chart(f2, use_container_width=True)

    st.markdown("#### Fault injection: ตรวจจับได้กี่ % ที่ระดับ fouling ต่างๆ")
    r2_ = np.random.default_rng(2)
    rows = []
    for sv_ in (0, 5, 10, 20, 30, 50):
        det = detect(inject(sl, sv_ / 100, vl0, vw0, ml["ua_med"], r2_), ml)
        rows.append({"UA ลดลง (%)": sv_, **{k: round(100 * float(np.mean(v)), 1) for k, v in det.items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    real_det = detect(sl, ml)
    fa = pd.DataFrame({"False alarm บนข้อมูลจริง (ครั้ง/สัปดาห์)": {k: round(float(np.sum(v)) / (len(sl) / 84), 1) for k, v in real_det.items()}})
    st.dataframe(fa, use_container_width=True)
    st.caption("ค่าในตารางคือ % ของจุดข้อมูลที่ถูกเตือน แถว 0% = false alarm บนข้อมูลสังเคราะห์ที่เครื่องปกติ "
               "(1 สัปดาห์ ≈ 84 จุด) — ใช้เทียบวิธีว่าวิธีไหนไวและเตือนผิดน้อยกว่า")

    st.markdown("#### เศรษฐศาสตร์การล้างเครื่อง (ตัวเลขต้นทุนเป็นค่าสมมติ ปรับได้)")
    e1, e2 = st.columns(2)
    c_clean = e1.number_input("ค่าล้างต่อครั้ง รวมเสียโอกาส (บาท)", 0.0, 1e7, 20000.0, 1000.0, key="c_clean")
    pen = e2.number_input("ต้นทุนความร้อนที่ระบายไม่ได้ (บาท/kWh)", 0.0, 100.0, 1.5, 0.1, key="pen")
    Td, dd = np.arange(5, 181), np.arange(0, 181)
    thm, tcm = float(sl.th_i.median()), float(sl.tc_i.median())
    q_clean = predict(thm, tcm, ml["ua_med"], vl0, vw0)[2]
    q_d = predict(thm, tcm, ml["ua_med"] * (1 - linf / 100 * (1 - np.exp(-dd / tau))), vl0, vw0)[2]
    cum = np.cumsum((q_clean - q_d) * 24 * pen)
    avg = (c_clean + cum[Td - 1]) / Td
    T_opt = int(Td[int(np.argmin(avg))])
    fe = go.Figure(go.Scatter(x=Td, y=avg, mode="lines", name="ต้นทุนเฉลี่ย/วัน"))
    fe.add_scatter(x=[T_opt], y=[float(avg.min())], mode="markers", marker_size=12, name="จุดคุ้มที่สุด")
    fe.update_layout(xaxis_title="รอบการล้าง (วัน)", yaxis_title="บาท/วัน", height=340)
    st.plotly_chart(fe, use_container_width=True)
    st.success(f"รอบล้างที่คุ้มที่สุดตามสมมติฐานนี้: ~{T_opt} วัน (ต้นทุนเฉลี่ย {avg.min():,.0f} บาท/วัน)")
    st.caption("ถ้าจุดต่ำสุดอยู่ที่ขอบ 180 วัน แปลว่าที่ fouling/ต้นทุนนี้ การล้างไม่คุ้ม — ผลขึ้นกับสมมติฐานทั้งหมดข้างบน")

# ============================================================
# แท็บ 5: Flow Sensitivity
# ============================================================
with tab5:
    st.subheader("Flow Sensitivity: เมื่อไม่มีอัตราการไหลจริง")
    st.markdown("#### 1) ประมาณ flow น้ำย้อนกลับจากสมดุลความร้อน (ให้ Lye flow ฐานเป็นหลัก)")
    est, er = {}, []
    for p in PHES:
        sp_ = M[p]["scored"]
        dtl, dtw = float((sp_.th_i - sp_.th_o).median()), float((sp_.tc_o - sp_.tc_i).median())
        est[p] = vl0 * RHO_L * CP_L * dtl / (RHO_W * CP_W * dtw) if dtw > 0 else np.nan
        er.append({"เครื่อง": p, "ΔT lye (°C)": round(dtl, 2), "ΔT น้ำ (°C)": round(dtw, 2),
                   f"Water flow ที่ทำให้สมดุล (m³/hr) เมื่อ Lye = {vl0}": round(est[p], 2)})
    st.dataframe(pd.DataFrame(er), hide_index=True, use_container_width=True)
    st.session_state["_vw_est"] = float(np.nanmean(list(est.values())))
    st.code("V_water / V_lye = (ρ_lye × Cp_lye × ΔT_lye) / (ρ_water × Cp_water × ΔT_water)", language="text")
    st.button("ใช้ค่าเฉลี่ยของสองเครื่องเป็น Cooling water flow ฐาน", on_click=cb_use_est)
    st.caption("เป็นการประมาณ: สมมติไม่มีความร้อนสูญเสีย และขึ้นกับ Lye flow ที่สมมติ — ใช้ตรวจความสอดคล้องเท่านั้น")

    st.markdown("#### 2) ข้อสรุปเปลี่ยนไหมเมื่อ flow เปลี่ยน")
    st.caption("การคูณ Lye flow ครอบคลุมความไม่แน่นอนของ ρ และ Cp ของ lye ด้วย (Cl = V × ρ × Cp) เช่น ±5% = แถว 0.95 / 1.05")
    sens = sens_table(raw, vl0, vw0)
    n_eps = int((sens["ε PHE#1"] > sens["ε PHE#2"]).sum())
    n_foul = int((((sens["p PHE#1"] < .05) & (sens["แนวโน้ม UA PHE#1 (%/วัน)"] < 0)) |
                  ((sens["p PHE#2"] < .05) & (sens["แนวโน้ม UA PHE#2 (%/วัน)"] < 0))).sum())
    s1, s2 = st.columns(2)
    s1.metric("ε ของ PHE#1 > PHE#2", f"{n_eps}/{len(sens)} กรณี")
    s2.metric("พบแนวโน้ม fouling", f"{n_foul}/{len(sens)} กรณี")
    fs_ = px.line(sens[sens["Water flow ×"] == 1.0], x="Lye flow ×", y=["UA PHE#1", "UA PHE#2"], markers=True)
    fs_.update_layout(yaxis_title="UA เฉลี่ย (W/K)", height=320)
    st.plotly_chart(fs_, use_container_width=True)
    with st.expander("ดูตารางทุกกรณี"):
        st.dataframe(sens, hide_index=True, use_container_width=True)

    st.markdown("#### 3) Counter-flow เทียบ Parallel-flow")
    pr = []
    for p in PHES:
        sp_ = D[D.phe == p].dropna(subset=["Ql"])
        d1, d2 = (sp_.th_i - sp_.tc_i).values, (sp_.th_o - sp_.tc_o).values
        with np.errstate(all="ignore"):
            lp = np.where((d1 > 0) & (d2 > 0) & (np.abs(d1 - d2) > 1e-6), (d1 - d2) / np.log(d1 / d2), np.nan)
        uap = np.nanmedian(sp_.Ql.values * 1000 / lp)
        pr.append({"เครื่อง": p, "UA counter-flow (W/K)": round(float(sp_.UA.median())),
                   "UA parallel-flow (W/K)": round(float(uap)), "อัตราส่วน": round(float(uap / sp_.UA.median()), 2)})
    st.dataframe(pd.DataFrame(pr), hide_index=True, use_container_width=True)
    st.caption("ถ้า PHE จริงเป็น counter-flow เกือบสมบูรณ์ (ปกติของ plate exchanger) ใช้ค่า counter-flow ได้ "
               "ส่วนอัตราส่วนบอกว่าสมมติฐานนี้เปลี่ยน UA มากแค่ไหน — แต่แนวโน้มเชิงเวลาไม่ขึ้นกับตรงนี้")

    st.markdown("#### 4) เปรียบเทียบ PHE#1 กับ PHE#2 ที่เงื่อนไขเดียวกัน (Lye 40 °C, น้ำ 30 °C)")
    cr = []
    for p in PHES:
        o = predict(40.0, 30.0, M[p]["ua_med"], vl0, vw0)
        cr.append({"เครื่อง": p, "UA (W/K)": round(M[p]["ua_med"]), "ε": round(float(o[3]), 3),
                   "Lye outlet (°C)": round(float(o[0]), 2), "Q (kW)": round(float(o[2]), 1)})
    st.dataframe(pd.DataFrame(cr), hide_index=True, use_container_width=True)
    st.caption("PHE#2 รับ lye ร้อนกว่า จึงเทียบ ΔT ตรงๆ ไม่ได้ — ตารางนี้ปรับเงื่อนไขให้เท่ากันก่อน")

    st.markdown("#### 5) Fouling resistance (Rf) — ต้องการข้อมูลจาก datasheet")
    if st.session_state.get("A_hx", 0) > 0 and st.session_state.get("U_clean", 0) > 0:
        rr = []
        for p in PHES:
            U = M[p]["ua_med"] / st.session_state.A_hx
            rr.append({"เครื่อง": p, "U (W/m²K)": round(U, 1),
                       "Rf (m²K/W)": f"{1 / U - 1 / st.session_state.U_clean:.5f}"})
        st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)
        st.code("U = UA / A\nRf = 1/U_fouled − 1/U_clean", language="text")
    else:
        st.info("กรอกพื้นที่ A และ U_clean ในแถบด้านซ้าย (กล่อง 'ข้อมูล datasheet') เมื่อมีข้อมูล ระบบจะคำนวณ Rf ให้ — "
                "ตอนนี้ใช้ Health Index = UA / UA_baseline เป็นตัวแทน")

# ============================================================
# แท็บ 6: โมเดล ข้อจำกัด และ Roadmap
# ============================================================
with tab6:
    st.subheader("Workflow")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box, style=rounded]; '
                      'A [label="Plant data (Excel)"]; B [label="Clean + QC"]; C [label="ฟิสิกส์: ε-NTU, UA, LMTD"]; '
                      'D [label="AI: RF / Hybrid / Isolation Forest"]; E [label="Simulator"]; F [label="Diagnostics + Fouling Lab"]; '
                      'A -> B -> C -> D; C -> E; D -> F; E -> F; }')
    st.subheader("สูตรหลักของแบบจำลอง")
    st.code("\n".join([
        "Cl = V_lye / 3600 × ρ_lye × Cp_lye",
        "Cw = V_water / 3600 × ρ_water × Cp_water",
        "Cmin = min(Cl, Cw)",
        "Cr = Cmin / Cmax",
        "NTU = UA / Cmin",
        "ε = (1 − exp(−NTU × (1 − Cr))) / (1 − Cr × exp(−NTU × (1 − Cr)))",
        "Q = ε × Cmin × (Th,in − Tc,in)",
        "Th,out = Th,in − Q / Cl",
        "Tc,out = Tc,in + Q / Cw",
        "UA = Q_lye / LMTD",
        "Health Index = UA / UA_baseline",
    ]), language="text")
    st.subheader("ข้อจำกัด (Limitations)")
    st.markdown("""
- ข้อมูลจริงเพียง ~14 วัน และเครื่องยังปกติ AI จึงเรียนรู้ "ลักษณะปกติ" ได้เท่านั้น
- อัตราการไหลเป็นค่าสมมติ ผลสัมบูรณ์ (kW, W/K) เป็นค่าประมาณ แต่ผลเชิงเปรียบเทียบและแนวโน้มทดสอบแล้วในแท็บ Flow Sensitivity
- ข้อมูลอ่านมือละเอียด 1 °C ทำให้ UA รายจุดไม่แน่นอนหลายสิบเปอร์เซ็นต์
- ไม่มี datasheet (พื้นที่, U_clean) จึงคำนวณ Rf จริงไม่ได้
- ข้อมูลไม่มีสถานะการเดินเครื่อง (load, start/stop) จึงแยก "ผิดปกติ" ออกจาก "เปลี่ยนโหลด" ไม่ได้
- ข้อมูลช่วง fouling ในแท็บ Fouling Lab และ Fault Classifier เป็นข้อมูลสังเคราะห์ ไม่ใช่ข้อมูลจริง (simulation-to-real ยังไม่ได้ยืนยัน)
- ΔP ในตัวจำลองและต้นทุนการล้างเป็นแบบจำลอง/ค่าสมมติที่ปรับได้
- เวอร์ชันนี้เป็น offline data-driven twin ยังไม่เชื่อม real-time
""")
    st.subheader("Roadmap")
    st.markdown("""
1. เก็บข้อมูลต่อเนื่องถึงวันล้างเครื่อง แล้วอัปโหลดหลายไฟล์พร้อมกัน (ระบบรวมและเทรนใหม่ให้)
2. ขอ flow จริงและ datasheet เพื่อคำนวณ Rf
3. เพิ่มคอลัมน์สถานะการเดินเครื่องในการบันทึก
4. เชื่อม sensor / PLC / historian (OPC-UA, MQTT) เพื่อเป็น twin แบบ real-time
""")


# ============================================================
# แท็บ 7: Fault Classifier (supervised จากข้อมูลสังเคราะห์)
# ============================================================
with tab7:
    st.subheader(f"Fault Classifier — {phe}")
    st.warning("โมเดลนี้เรียนรู้จาก **ข้อมูลสังเคราะห์** (ฟิสิกส์ + อุณหภูมิขาเข้าจริง + noise ระดับเดียวกับข้อมูลจริง) "
               "เพราะข้อมูลจริงไม่มีตัวอย่างความผิดปกติ — ความแม่นยำด้านล่างวัดบนข้อมูลสังเคราะห์ ยังไม่ได้ยืนยันกับเครื่องจริง")
    cl_ = build_clf(M[phe]["scored"], M[phe]["ua_med"], vl0, vw0)
    z1, z2, z3 = st.columns(3)
    z1.metric("Accuracy (ข้อมูลสังเคราะห์ที่ไม่เคยเห็น)", f"{cl_['acc'] * 100:.1f} %")
    z2.metric("ข้อมูลจริงที่ถูกจัดเป็น Normal", f"{cl_['dist']['Normal']:.1f} %")
    z3.metric("Noise ที่ใช้ (σ lye / σ น้ำ)", f"{cl_['sig'][0]:.2f} / {cl_['sig'][1]:.2f} °C")

    st.markdown("#### สถานะที่ตั้งไว้ในตัวจำลอง (แท็บ 2) ถูกจัดเป็นอะไร")
    sim_x = pd.DataFrame([[th - cl_["base"][0], tc - cl_["base"][1], th_o - b_tho, tc_o - b_tco]], columns=CF)
    lab_, pp_ = decide(cl_["clf"], sim_x)
    pr_ = pd.Series(pp_[0], index=cl_["clf"].classes_).reindex(CLASSES) * 100
    st.success(f"ผลจัดประเภท: **{lab_[0]}** (ความน่าจะเป็นสูงสุด {pr_.max():.0f}%; ถือเป็นความผิดปกติเมื่อ ≥ 60%)")
    fp = px.bar(pr_.round(1).reset_index().rename(columns={"index": "ประเภท", 0: "ความน่าจะเป็น (%)"}),
                x="ประเภท", y="ความน่าจะเป็น (%)")
    fp.update_layout(height=300)
    st.plotly_chart(fp, use_container_width=True)
    st.caption("ลองเลือก preset ในแถบด้านซ้าย (เช่น Lye Flow Drop) แล้วดูว่าโมเดลแยกอาการได้หรือไม่")

    g1, g2 = st.columns(2)
    fc = px.imshow(cl_["cm"], text_auto=True, aspect="auto", labels=dict(x="ทำนาย", y="จริง", color="จำนวน"))
    fc.update_layout(height=420, title="Confusion matrix (ข้อมูลสังเคราะห์ชุดทดสอบ)")
    g1.plotly_chart(fc, use_container_width=True)
    g2.markdown("**Recall ของ Fouling ตามความรุนแรง**")
    g2.dataframe(cl_["rec"], hide_index=True, use_container_width=True)
    g2.markdown("**ความสำคัญของ feature**")
    g2.dataframe(pd.DataFrame({"importance": cl_["imp"]}), use_container_width=True)

    st.markdown("#### ข้อมูลจริงถูกจัดเป็นอะไรบ้าง (ควรเป็น Normal เป็นส่วนใหญ่)")
    ss_ = M[phe]["scored"]
    fr_ = px.scatter(pd.DataFrame({"Time": ss_.Time, "ประเภท": cl_["real_pred"]}), x="Time", y="ประเภท",
                     category_orders={"ประเภท": CLASSES})
    fr_.update_layout(height=320)
    st.plotly_chart(fr_, use_container_width=True)
    st.dataframe(cl_["dist"].rename("% ของข้อมูลจริง").to_frame(), use_container_width=True)
    st.caption("ที่ถูกจัดเป็นความผิดปกติในข้อมูลจริง = false alarm ของ classifier (ข้อมูลจริงช่วงนี้ไม่มีเหตุการณ์ผิดปกติที่ยืนยันได้) "
               "ใช้ cross-validation แบบแบ่งบล็อกเพื่อคำนวณความเบี่ยงเบนจากค่าปกติ")


# ============================================================
# แท็บ 0: Overview (ไฟจราจร) + Demo
# ============================================================
STATUS = {0: ("#2E8B57", "🟢 ปกติ"), 1: ("#C45A10", "🟡 เฝ้าระวัง"), 2: ("#A50000", "🔴 ควรวางแผนล้างเครื่อง")}


def banner(title, level, size=26):
    c, label = STATUS[level]
    st.markdown(f"<div style='background:{c};color:white;padding:16px;border-radius:12px;font-size:{size}px;"
                f"font-weight:700;text-align:center'>{title}: {label}</div>", unsafe_allow_html=True)


def health_status(s, base_n=60, recent_n=24):
    """ประเมินสุขภาพจากข้อมูลจริง: UA ล่าสุดเทียบช่วงแรก + แนวโน้ม + จุดผิดปกติล่าสุด"""
    base, rec = s.UA.iloc[:base_n].median(), s.UA.iloc[-recent_n:].median()
    hi = float(rec / base)
    lr = linregress(s.day, s.UA)
    slope, p = float(lr.slope / s.UA.median() * 100), float(lr.pvalue)
    flag = float(s.anom.iloc[-recent_n:].mean())
    level = 2 if (hi < 0.70 or (p < 0.05 and slope < -2)) else 1 if (hi < 0.85 or (p < 0.05 and slope < 0) or flag > 0.3) else 0
    return dict(hi=hi, slope=slope, p=p, flag=flag, level=level, base=float(base),
                dp_rec=float(s.dP_l.iloc[-recent_n:].median()), dp_base=float(s.dP_l.iloc[:base_n].median()))


def make_syn(ml, sl, days, linf, tau, vl, vw, seed=1):
    r = np.random.default_rng(seed)
    n = days * 12
    idx = r.integers(0, len(sl), n)
    t = np.arange(n) / 12
    loss = linf / 100 * (1 - np.exp(-t / tau))
    thi, tci = sl.th_i.values[idx], sl.tc_i.values[idx]
    tho, tco = predict(thi, tci, ml["ua_med"] * (1 - loss), vl, vw)[:2]
    tho = np.round(tho + r.normal(0, .3, n))
    tco = np.round(tco + r.normal(0, .3, n))
    cc = calc(thi, tho, tci, tco, vl, vw)
    out = pd.DataFrame({"t": t, "UA": cc["UA"], "th_o": tho, "loss": loss})
    out["HI"] = out.UA.rolling(24, min_periods=12).median() / ml["ua_med"]
    return out


with tab0:
    st.subheader("ภาพรวมสุขภาพเครื่องแลกเปลี่ยนความร้อน")
    st.caption(f"คำนวณจากข้อมูลจริงถึง {raw.Time.max():%d %b %Y %H:%M} — เทียบ UA ล่าสุด (1 วัน) กับช่วงแรกของข้อมูล")
    oc = st.columns(2)
    for col, p in zip(oc, PHES):
        sp_ = M[p]["scored"]
        h_ = health_status(sp_)
        with col:
            banner(p, h_["level"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Health Index ล่าสุด", f"{h_['hi'] * 100:.0f} %")
            k2.metric("แนวโน้ม UA", f"{h_['slope']:+.2f} %/วัน", f"p = {h_['p']:.2f}", delta_color="off")
            k3.metric("ΔP lye ล่าสุด", f"{h_['dp_rec']:.2f}", f"{h_['dp_rec'] - h_['dp_base']:+.2f} จากช่วงแรก", delta_color="inverse")
            advice = {0: "ทำงานปกติ ไม่พบแนวโน้ม fouling — เฝ้าระวังต่อตามรอบ",
                      1: f"UA ต่ำกว่าช่วงแรกประมาณ {(1 - h_['hi']) * 100:.0f}% หรือมีแนวโน้มลดลง — ตรวจ ΔP และเตรียมแผนล้าง",
                      2: "การระบายความร้อนลดลงมาก — ควรวางแผนล้างเครื่องโดยเร็ว"}[h_["level"]]
            st.info(advice)
            hi_s = sp_.UA.rolling(24, min_periods=12).median() / h_["base"]
            fo = go.Figure(go.Scatter(x=sp_.Time, y=hi_s, mode="lines", name="Health Index"))
            fo.add_hline(y=0.85, line_dash="dot", annotation_text="เตือน")
            fo.add_hline(y=0.70, line_dash="dot", annotation_text="ล้าง")
            fo.update_layout(height=260, yaxis_tickformat=".0%", yaxis_range=[0.4, 1.3], margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fo, use_container_width=True)
    with st.expander("ระบบตัดสินสถานะอย่างไร"):
        st.markdown("- 🔴 Health Index < 70% หรือ UA ลดลงเกิน 2%/วันอย่างมีนัยสำคัญ\n"
                    "- 🟡 Health Index < 85%, มีแนวโน้มลดลงอย่างมีนัยสำคัญ, หรือจุดผิดปกติเกิน 30% ในวันล่าสุด\n"
                    "- 🟢 นอกเหนือจากนั้น\n\n"
                    "Health Index = UA ล่าสุด (median 1 วัน) ÷ UA ช่วงแรกของข้อมูล (60 จุดแรก)")

    st.divider()
    st.markdown("### ▶ Demo: เล่นสถานการณ์ fouling สะสม (ข้อมูลสังเคราะห์)")
    st.caption("จำลองเครื่องที่สกปรกขึ้นทีละวัน ดูว่าระบบเตือนวันไหนเทียบกับวันที่ควรเตือนจริง")
    d1, d2, d3, d4 = st.columns(4)
    dphe = d1.radio("เครื่อง", PHES, key="demo_phe", horizontal=True)
    dd_ = d2.slider("จำนวนวัน", 20, 90, 45, key="demo_days")
    dl_ = d3.slider("UA ลดลงสุดท้าย (%)", 10, 70, 45, key="demo_loss")
    sp_s = d4.select_slider("ความเร็ว", ["ช้า", "กลาง", "เร็ว"], value="กลาง", key="demo_speed")
    if st.button("▶ เริ่ม Demo", type="primary", key="demo_go"):
        mm_ = M[dphe]
        syn_ = make_syn(mm_, mm_["scored"], dd_, dl_, max(dd_ / 2.5, 5), vl0, vw0)
        delay, ph_ = {"ช้า": 0.35, "กลาง": 0.15, "เร็ว": 0.04}[sp_s], st.empty()
        wd_ = cd_ = None
        for day in range(1, dd_ + 1):
            cur = syn_[syn_.t < day]
            hi_ = float(cur.HI.dropna().iloc[-1]) if cur.HI.notna().any() else 1.0
            lv_ = 2 if hi_ < 0.70 else 1 if hi_ < 0.85 else 0
            wd_ = day if (lv_ >= 1 and wd_ is None) else wd_
            cd_ = day if (lv_ == 2 and cd_ is None) else cd_
            with ph_.container():
                banner(f"{dphe} · วันที่ {day}", lv_, 22)
                q1, q2, q3 = st.columns(3)
                q1.metric("Health Index (ระบบวัดได้)", f"{hi_ * 100:.0f} %")
                q2.metric("UA ลดลงจริงในแบบจำลอง", f"{cur.loss.iloc[-1] * 100:.0f} %")
                q3.metric("Lye outlet เฉลี่ย 1 วัน", f"{cur.th_o.iloc[-12:].mean():.1f} °C")
                fd = go.Figure()
                fd.add_scatter(x=cur.t, y=cur.HI, name="Health Index (วัดได้)")
                fd.add_scatter(x=cur.t, y=1 - cur.loss, name="ค่าจริงในแบบจำลอง", line_dash="dash")
                fd.add_hline(y=0.85, line_dash="dot", annotation_text="เตือน")
                fd.add_hline(y=0.70, line_dash="dot", annotation_text="ล้าง")
                fd.update_layout(height=320, xaxis_range=[0, dd_], yaxis_range=[0.4, 1.15], yaxis_tickformat=".0%",
                                 xaxis_title="วัน", margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(fd, use_container_width=True)
            time.sleep(delay)
        tw = next((d for d, l in zip(syn_.t, 1 - syn_.loss) if l < 0.85), None)
        tcl = next((d for d, l in zip(syn_.t, 1 - syn_.loss) if l < 0.70), None)
        f_ = lambda v: "ไม่ถึงเกณฑ์" if v is None else f"วันที่ {v:.0f}"
        st.success(f"สรุป: ระบบเตือนครั้งแรก {f_(wd_)} (จริงควรเตือน {f_(tw)}) · ถึงเกณฑ์ล้าง {f_(cd_)} (จริง {f_(tcl)})")
        st.caption("ข้อมูลสังเคราะห์จากแบบจำลองฟิสิกส์ + อุณหภูมิขาเข้าจริง ไม่ใช่ข้อมูล fouling จริงของโรงงาน")
