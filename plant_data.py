import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import IsolationForest

RHO_LYE, CP_LYE = 1450.0, 3200.0      # NaOH 43%
RHO_WTR, CP_WTR = 996.0, 4180.0
V_LYE_ASSUMED, V_WTR_ASSUMED = 7.0, 10.0   # m3/h
RES_LYE, RES_WTR = 1.0, 0.1   # ความละเอียดการอ่าน (°C)

COLS = ["Time",
        "P1_Tw_in", "P1_Tw_out", "P1_Tl_in", "P1_Tl_out",
        "P2_Tw_in", "P2_Tw_out", "P2_Tl_in", "P2_Tl_out",
        "P1_Pw_in", "P1_Pw_out", "P1_Pl_in", "P1_Pl_out",
        "P2_Pw_in", "P2_Pw_out", "P2_Pl_in", "P2_Pl_out"]

MANUAL_EVENTS = pd.to_datetime([
    "2026-06-05 18:00",
    "2026-06-12 02:00",
    "2026-06-06 14:00",
])

def load_raw(file, sheet="Raw Data"):
    raw = pd.read_excel(file, sheet_name=sheet, header=None).iloc[:, :17]
    raw.columns = COLS
    t = pd.to_datetime(raw["Time"], format="%d %b %y %H:%M", errors="coerce")
    raw["Time"] = t.fillna(pd.to_datetime(raw["Time"], errors="coerce"))
    df = raw.dropna(subset=["Time"]).copy()
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=COLS[1:9]).reset_index(drop=True)

def cols(p):
    return f"{p}_Tl_in", f"{p}_Tl_out", f"{p}_Tw_in", f"{p}_Tw_out"

def quality_flags(df, p):
    thi, tho, tci, tco = cols(p)
    f = pd.DataFrame({"Time": df["Time"]})
    f["P_water_reversed"] = df[f"{p}_Pw_out"] > df[f"{p}_Pw_in"]
    f["P_lye_reversed"] = df[f"{p}_Pl_out"] > df[f"{p}_Pl_in"]
    f["T_out_of_range"] = ~df[[thi, tho, tci, tco]].apply(lambda s: s.between(10, 80)).all(axis=1)
    f["T_2nd_law_violation"] = (df[tco] > df[thi]) | (df[tho] < df[tci])
    f["any_flag"] = f.drop(columns="Time").any(axis=1)
    return f

def predict_outlets(Thi, Tci, C_lye, C_wtr, UA):
    Cmin, Cmax = min(C_lye, C_wtr), max(C_lye, C_wtr)
    Cr, NTU = Cmin / Cmax, UA / Cmin
    if abs(1 - Cr) < 1e-6:
        eps = NTU / (1 + NTU)
    else:
        e = np.exp(-NTU * (1 - Cr))
        eps = (1 - e) / (1 - Cr * e)
    Q = eps * Cmin * (Thi - Tci)
    return Thi - Q / C_lye, Tci + Q / C_wtr, Q

def calibrate(train, p, C_wtr=(V_WTR_ASSUMED/3600*RHO_WTR*CP_WTR)):
    thi, tho, tci, tco = cols(p)
    Thi, Tci = train[thi].values, train[tci].values
    Tho_m, Tco_m = train[tho].values, train[tco].values
    C_lye_init = V_LYE_ASSUMED/3600 * RHO_LYE * CP_LYE

    def resid(x):
        UA, C_lye = x
        Tho, Tco, _ = predict_outlets(Thi, Tci, C_lye, C_wtr, UA)
        return np.r_[Tho - Tho_m, Tco - Tco_m]

    res = least_squares(resid, x0=[5000.0, C_lye_init],
                        bounds=([100, 500], [1e5, 1e5]), loss="soft_l1")
    UA, C_lye = res.x
    return {"UA": UA, "C_lye": C_lye, "V_lye_implied": C_lye / (RHO_LYE * CP_LYE) * 3600}

def validate(df, p, n_train, clean_mask):
    thi, tho, tci, tco = cols(p)
    train = df.iloc[:n_train][clean_mask.iloc[:n_train].values]
    test = df.iloc[n_train:]
    par = calibrate(train, p)
    Tho, Tco, _ = predict_outlets(test[thi].values, test[tci].values,
                                  par["C_lye"], (V_WTR_ASSUMED/3600*RHO_WTR*CP_WTR), par["UA"])
    eh, ec = test[tho].values - Tho, test[tco].values - Tco
    nh = test[tho].values - train[tho].mean()
    nc = test[tco].values - train[tco].mean()
    metrics = pd.DataFrame({
        "Variable": ["Lye outlet", "Water outlet"],
        "MAE model (°C)": [np.abs(eh).mean(), np.abs(ec).mean()],
        "RMSE model (°C)": [np.sqrt((eh**2).mean()), np.sqrt((ec**2).mean())],
        "MAE naive (°C)": [np.abs(nh).mean(), np.abs(nc).mean()],
        "Sensor resolution (°C)": [RES_LYE, RES_WTR],
    })
    pred = pd.DataFrame({"Time": test["Time"].values,
                         "Lye_meas": test[tho].values, "Lye_pred": Tho,
                         "Wtr_meas": test[tco].values, "Wtr_pred": Tco})
    return par, metrics, pred

FEATS = ["eps_wtr", "approach_hot", "approach_cold", "LMTD",
         "dP_wtr", "dP_lye", "res_lye", "res_wtr"]

def features(df, p, par):
    thi, tho, tci, tco = cols(p)
    dTh, dTc = df[thi] - df[tho], df[tco] - df[tci]
    dT1, dT2 = df[thi] - df[tco], df[tho] - df[tci]
    with np.errstate(divide="ignore", invalid="ignore"):
        lmtd = np.where(np.isclose(dT1, dT2), dT1, (dT1 - dT2) / np.log(dT1 / dT2))
    lmtd = np.where((dT1 > 0) & (dT2 > 0), lmtd, np.nan)

    Ch, Cc = par["C_lye"], (V_WTR_ASSUMED/3600*RHO_WTR*CP_WTR)
    sig = np.sqrt((Ch * np.sqrt(2) * RES_LYE / np.sqrt(12))**2 +
                  (Cc * np.sqrt(2) * RES_WTR / np.sqrt(12))**2)
    Tho_p, Tco_p, _ = predict_outlets(df[thi].values, df[tci].values, Ch, Cc, par["UA"])
    m = pd.DataFrame({
        "Time": df["Time"],
        "eps_wtr": dTc / (df[thi] - df[tci]),
        "approach_hot": dT1, "approach_cold": dT2, "LMTD": lmtd,
        "dP_wtr": df[f"{p}_Pw_in"] - df[f"{p}_Pw_out"],
        "dP_lye": df[f"{p}_Pl_in"] - df[f"{p}_Pl_out"],
        "res_lye": df[tho] - Tho_p,
        "res_wtr": df[tco] - Tco_p,
        "EB_z": np.abs(Ch * dTh - Cc * dTc) / sig,
    })
    return m.replace([np.inf, -np.inf], np.nan)

def detect(m, n_train, clean_mask, contamination=0.05):
    X = m[FEATS].fillna(m[FEATS].median())
    train_X = X.iloc[:n_train][clean_mask.iloc[:n_train].values]
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42).fit(train_X)
    res = m.copy()
    res["score"] = iso.score_samples(X)
    res["AI_anomaly"] = iso.predict(X) == -1
    return res

def compare_with_engineer(m, flags):
    eng = flags["any_flag"].values | m["Time"].isin(MANUAL_EVENTS).values
    return pd.crosstab(pd.Series(eng, name="Engineer / Rule Flag"),
                       pd.Series(m["AI_anomaly"].values, name="AI Anomaly Flag"))
# ---------- 7. Summary Insights for Dashboard ----------
def get_plant_summary(m, flags, unit_name):
    total_points = len(m)
    quality_issues = int(flags["any_flag"].sum())
    ai_anomalies = int(m["AI_anomaly"].sum())
    avg_lmtd = m["Lmtd" if "Lmtd" in m.columns else "LMTD"].mean()
    
    summary_df = pd.DataFrame({
        "Metric": [
            "Equipment Monitored",
            "Total Operating Records",
            "Data Quality Flags (Sensor/Physical checks)",
            "AI Detected Anomalies",
            "Average LMTD (°C)"
        ],
        "Value": [
            f"Plate Heat Exchanger ({unit_name})",
            f"{total_points} records (14 days)",
            f"{quality_issues} points ({quality_issues/total_points*100:.1f}%)",
            f"{ai_anomalies} points",
            f"{avg_lmtd:.2f} °C"
        ]
    })
    return summary_df
    