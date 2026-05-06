"""
Compute Avg American axis values from NHANES microdata.

Outputs a JSON with: risk_avg, reserve_avg, lifestyle_avg
plus component-level summaries showing which percentile of each
the median American sits at.

All percentiles are computed against the NHANES adult (20+) distribution
itself. So "Avg American" is, by construction, percentile ~50 within
NHANES on each component, and the composite numbers fall out from
the directional flips and weighting.

Reserve example: median American grip strength is by definition the
50th percentile of NHANES grip strength. But the composite uses
"better-is-higher" coding, and the population median anchors below
the reference of "fit" (which we calibrate against population top quartile)
because the composite's 0-100 scale is anchored to the realistic human
range, not to NHANES median.

For the prototype's slider, we want each axis on the same 0-100 scale
where:
  0 = practical floor (severe deconditioning / chronic poor / optimal biomarkers)
  50 = midpoint of population distribution
  100 = practical ceiling (elite / optimal felt / highest realistic risk)

We map each NHANES median directly to its percentile in NHANES (which
is by definition 50 if the variable is symmetric around its median).
But "avg American" on the SLIDER is the percentile re-anchored against
the realistic human range, not against NHANES.

For v2, we use a pragmatic approach:
  - For each component, compute NHANES median raw value
  - Compare to published "elite" or "ideal" benchmark
  - Express as 0-100 where 100 = elite/ideal benchmark, 0 = floor
  - Weighted composite per the spec

This is documented per-component in the code below.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt

OUT = Path(__file__).parent / "calibration_outputs"
OUT.mkdir(exist_ok=True)

ADULT_AGE_MIN = 20
ADULT_AGE_MAX = 79  # PREVENT validity range


def adults(df, age_col="RIDAGEYR"):
    return df[(df[age_col] >= ADULT_AGE_MIN) & (df[age_col] <= ADULT_AGE_MAX)]


# ---------- RISK COMPONENTS (NHANES 2017-2020) ----------

def risk_components():
    demo = read_xpt("DEMO_P")
    bpxo = read_xpt("BPXO_P")
    bmx  = read_xpt("BMX_P")
    tchol = read_xpt("TCHOL_P")
    hdl  = read_xpt("HDL_P")
    trigly = read_xpt("TRIGLY_P")  # has LBDLDL (LDL)
    ghb  = read_xpt("GHB_P")
    smq  = read_xpt("SMQ_P")
    diq  = read_xpt("DIQ_P")
    bpq  = read_xpt("BPQ_P")
    biopro = read_xpt("BIOPRO_P")  # has LBXSCR (creatinine)

    # Adult subset
    a = adults(demo[["SEQN","RIDAGEYR","RIAGENDR"]])
    a = a.merge(bpxo[["SEQN","BPXOSY1","BPXOSY2","BPXOSY3","BPXODI1","BPXODI2","BPXODI3"]], on="SEQN", how="left")
    a = a.merge(bmx[["SEQN","BMXBMI"]], on="SEQN", how="left")
    a = a.merge(tchol[["SEQN","LBXTC"]], on="SEQN", how="left")
    a = a.merge(hdl[["SEQN","LBDHDD"]], on="SEQN", how="left")
    a = a.merge(trigly[["SEQN","LBDLDL"]], on="SEQN", how="left")
    a = a.merge(ghb[["SEQN","LBXGH"]], on="SEQN", how="left")
    a = a.merge(diq[["SEQN","DIQ010"]], on="SEQN", how="left")  # 1=yes diabetes
    # Smoking: SMQ020 (ever 100 cigs) + SMQ040 (now smoke)
    a = a.merge(smq[["SEQN","SMQ020","SMQ040"]], on="SEQN", how="left")
    # BP/chol meds
    a = a.merge(bpq[["SEQN","BPQ050A","BPQ100D"]], on="SEQN", how="left") if "BPQ050A" in bpq.columns else a
    a = a.merge(biopro[["SEQN","LBXSCR"]], on="SEQN", how="left")

    # Mean BP from up to 3 readings
    sbp_cols = ["BPXOSY1","BPXOSY2","BPXOSY3"]
    dbp_cols = ["BPXODI1","BPXODI2","BPXODI3"]
    a["SBP"] = a[sbp_cols].mean(axis=1, skipna=True)
    a["DBP"] = a[dbp_cols].mean(axis=1, skipna=True)
    # Current smoker: SMQ040 in (1,2) = every day / some days
    a["smoker"] = a["SMQ040"].isin([1, 2]).astype(int)
    # Diabetes: DIQ010 == 1
    a["diabetes"] = (a["DIQ010"] == 1).astype(int)
    # eGFR from creatinine (CKD-EPI 2021 race-free):
    # Simplified: 142 * min(Scr/k, 1)^a * max(Scr/k, 1)^-1.200 * 0.9938^age * (1.012 if female)
    # k = 0.7 (F), 0.9 (M); a = -0.241 (F), -0.302 (M)
    def egfr(row):
        scr = row["LBXSCR"]
        if pd.isna(scr): return np.nan
        age = row["RIDAGEYR"]
        female = row["RIAGENDR"] == 2
        k = 0.7 if female else 0.9
        alpha = -0.241 if female else -0.302
        sex_mult = 1.012 if female else 1.0
        ratio = scr/k
        e = 142 * (min(ratio,1)**alpha) * (max(ratio,1)**-1.200) * (0.9938**age) * sex_mult
        return e
    a["eGFR"] = a.apply(egfr, axis=1)

    summary = {
        "n_adults": int(len(a)),
        "median_age": float(a["RIDAGEYR"].median()),
        "TC_mg_dL_median":     float(a["LBXTC"].median(skipna=True)),
        "HDL_mg_dL_median":    float(a["LBDHDD"].median(skipna=True)),
        "LDL_mg_dL_median":    float(a["LBDLDL"].median(skipna=True)),
        "SBP_mmHg_median":     float(a["SBP"].median(skipna=True)),
        "DBP_mmHg_median":     float(a["DBP"].median(skipna=True)),
        "BMI_median":          float(a["BMXBMI"].median(skipna=True)),
        "A1c_pct_median":      float(a["LBXGH"].median(skipna=True)),
        "eGFR_median":         float(a["eGFR"].median(skipna=True)),
        "current_smoker_pct":  float(a["smoker"].mean()*100),
        "diabetes_pct":        float(a["diabetes"].mean()*100),
        # prevalence anchors
        "BMI_obese_pct":       float((a["BMXBMI"] >= 30).mean()*100),
        "BMI_overweight_or_obese_pct": float((a["BMXBMI"] >= 25).mean()*100),
        "SBP_hypertension_pct":float((a["SBP"] >= 130).mean()*100),
        "LDL_high_pct":        float((a["LBDLDL"] >= 130).mean()*100),
        "A1c_prediabetes_pct": float(((a["LBXGH"] >= 5.7) & (a["LBXGH"] < 6.5)).mean()*100),
        "A1c_diabetes_pct":    float((a["LBXGH"] >= 6.5).mean()*100),
    }
    return summary, a


# ---------- RESERVE COMPONENTS ----------

def reserve_components():
    demo_h = read_xpt("DEMO_H")
    mgx_h  = read_xpt("MGX_H")
    demo_g = read_xpt("DEMO_G")
    mgx_g  = read_xpt("MGX_G")
    cfq_g  = read_xpt("CFQ_G")
    demo_c = read_xpt("DEMO_C")
    cvx_c  = read_xpt("CVX_C")

    # Grip strength: combine 2011-2014 (G + H), use combined max grip across hands
    parts = []
    for demo, mgx in [(demo_g, mgx_g), (demo_h, mgx_h)]:
        a = adults(demo[["SEQN","RIDAGEYR","RIAGENDR"]])
        a = a.merge(mgx, on="SEQN", how="inner")
        # MGXH1T1, MGXH1T2, MGXH1T3 = right hand kg trials. MGXH2T1.. = left hand.
        right = a[["MGXH1T1","MGXH1T2","MGXH1T3"]].max(axis=1, skipna=True)
        left  = a[["MGXH2T1","MGXH2T2","MGXH2T3"]].max(axis=1, skipna=True)
        a["grip_max_kg"] = pd.concat([right, left], axis=1).max(axis=1, skipna=True)
        a["combined_grip_kg"] = right.fillna(0) + left.fillna(0)
        parts.append(a[["RIDAGEYR","RIAGENDR","grip_max_kg","combined_grip_kg"]].dropna(subset=["grip_max_kg"]))
    grip_df = pd.concat(parts, ignore_index=True)

    # DSST (cognitive) — CFDDS = digit symbol substitution score
    a_cog = adults(demo_g[["SEQN","RIDAGEYR","RIAGENDR"]])
    a_cog = a_cog.merge(cfq_g[["SEQN","CFDDS"]], on="SEQN", how="inner").dropna(subset=["CFDDS"])

    # VO2max submax — 2003-2004 had submax test estimating VO2max
    # CVDVOMAX = predicted VO2max (mL/kg/min) from submax test
    a_vo2 = adults(demo_c[["SEQN","RIDAGEYR","RIAGENDR"]])
    if "CVDVOMAX" in cvx_c.columns:
        a_vo2 = a_vo2.merge(cvx_c[["SEQN","CVDVOMAX"]], on="SEQN", how="inner").dropna(subset=["CVDVOMAX"])
    else:
        a_vo2 = pd.DataFrame()

    summary = {
        "grip_n": int(len(grip_df)),
        "grip_max_kg_median":   float(grip_df["grip_max_kg"].median()),
        "grip_combined_kg_median": float(grip_df["combined_grip_kg"].median()),
        "grip_male_max_kg_median":   float(grip_df.loc[grip_df["RIAGENDR"]==1,"grip_max_kg"].median()),
        "grip_female_max_kg_median": float(grip_df.loc[grip_df["RIAGENDR"]==2,"grip_max_kg"].median()),
        "DSST_n": int(len(a_cog)),
        "DSST_median": float(a_cog["CFDDS"].median()),
        "VO2max_n": int(len(a_vo2)),
        "VO2max_mL_kg_min_median": float(a_vo2["CVDVOMAX"].median()) if len(a_vo2) else None,
    }
    return summary, grip_df, a_cog, a_vo2


# ---------- LIFESTYLE COMPONENTS ----------

def lifestyle_components():
    demo = read_xpt("DEMO_P")
    dpq = read_xpt("DPQ_P")  # PHQ-9
    slq = read_xpt("SLQ_P")  # sleep duration

    a = adults(demo[["SEQN","RIDAGEYR"]])

    # PHQ-9: DPQ010..DPQ090, sum (0-3 each, total 0-27). Skip DPQ100 (functional impact).
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    if all(c in dpq.columns for c in phq_cols):
        d = a.merge(dpq[["SEQN"] + phq_cols], on="SEQN", how="inner")
        # Recode 7,9 (refused/don't know) as NaN
        for c in phq_cols:
            d[c] = d[c].where(d[c].isin([0,1,2,3]), np.nan)
        d["PHQ9_total"] = d[phq_cols].sum(axis=1, skipna=False)
        d = d.dropna(subset=["PHQ9_total"])
    else:
        d = pd.DataFrame()

    # Sleep: SLD012 = hours weekday, SLD013 = hours weekend. Use weekday average.
    s = a.merge(slq[["SEQN","SLD012","SLD013"]], on="SEQN", how="inner") if "SLD012" in slq.columns else pd.DataFrame()
    if len(s):
        s = s.dropna(subset=["SLD012"])
        s["short_sleep"] = (s["SLD012"] < 7).astype(int)

    summary = {
        "PHQ9_n": int(len(d)),
        "PHQ9_median": float(d["PHQ9_total"].median()) if len(d) else None,
        "PHQ9_mean":   float(d["PHQ9_total"].mean()) if len(d) else None,
        "PHQ9_moderate_or_severe_pct": float((d["PHQ9_total"] >= 10).mean()*100) if len(d) else None,
        "PHQ9_any_depression_pct":     float((d["PHQ9_total"] >= 5).mean()*100) if len(d) else None,
        "Sleep_n": int(len(s)),
        "Sleep_hours_median": float(s["SLD012"].median()) if len(s) else None,
        "Sleep_short_pct": float(s["short_sleep"].mean()*100) if len(s) else None,
    }
    return summary, d, s


# ---------- COMPOSITE SCORING ----------

def compute_avg_american(risk_s, reserve_s, lifestyle_s):
    """
    Map measured medians to 0-100 axis positions per the calibration spec.

    Approach: for each component, define an "ideal" anchor (= 100) and
    a "floor" anchor (= 0) from clinical/literature norms, then linearly
    interpolate the NHANES median into that range.
    """

    # ===== RISK =====
    # Risk composite: higher is worse. We compute a 0-100 risk score where
    # each component is mapped using clinical thresholds.
    # Anchors (100 = "high" risk, 0 = "optimal" risk):
    #   LDL: 0=70, 100=190
    #   SBP: 0=110, 100=160
    #   A1c: 0=5.0, 100=8.0
    #   BMI: 0=22, 100=40
    #   smoking_pct: literally the prevalence among the cohort (already 0-100)
    #   diabetes_pct: same
    # Then weight equally for v2 simplicity. Future: PREVENT-derived.

    def lerp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))

    risk_components_scored = {
        "LDL":      lerp(risk_s["LDL_mg_dL_median"], 70, 190),
        "SBP":      lerp(risk_s["SBP_mmHg_median"], 110, 160),
        "A1c":      lerp(risk_s["A1c_pct_median"], 5.0, 8.0),
        "BMI":      lerp(risk_s["BMI_median"], 22, 40),
        # For prevalences: convert raw % into 0-100 directly (already on that scale, just clip)
        "smoking_prevalence":  float(np.clip(risk_s["current_smoker_pct"]*5, 0, 100)),  # 20% smoking → 100
        "diabetes_prevalence": float(np.clip(risk_s["diabetes_pct"]*5, 0, 100)),
    }
    # Equal weights for v2, except smoking/diabetes which act as multipliers
    # In a real PREVENT-residualized model these would be unified.
    risk_avg = float(np.mean(list(risk_components_scored.values())))

    # ===== RESERVE =====
    # Reserve composite: higher is better. Avg American gets percentile ~50
    # of NHANES, but on the slider that anchors against population midpoint.
    # We use literature-derived "elite" and "deconditioning" cutpoints:
    #   VO2max:     0 = 18 mL/kg/min, 100 = 55
    #   grip_max:   0 = 20 kg,        100 = 60
    #   DSST:       0 = 25 correct,   100 = 75
    # Compose with declared weights (cardiorespiratory dominant).

    res = {
        "VO2max":   lerp(reserve_s["VO2max_mL_kg_min_median"], 18, 55) if reserve_s.get("VO2max_mL_kg_min_median") else None,
        "grip_max": lerp(reserve_s["grip_max_kg_median"], 20, 60),
        "DSST":     lerp(reserve_s["DSST_median"], 25, 75),
    }
    # Weights: VO2max 50, grip 25, DSST 25 (collapsed from full 8-component)
    if res["VO2max"] is not None:
        reserve_avg = 0.50 * res["VO2max"] + 0.25 * res["grip_max"] + 0.25 * res["DSST"]
    else:
        reserve_avg = 0.5 * res["grip_max"] + 0.5 * res["DSST"]

    # ===== LIFESTYLE =====
    # Lifestyle composite: higher = better. Derived from PHQ-9 (depression)
    # and sleep hours; PSQI not in NHANES so we use sleep duration as a proxy.
    #   PHQ-9: 0=27 (worst), 100=0 (no symptoms)
    #   sleep_hours: 0=4 hrs, 100=8 hrs (capped both ends)
    life_phq = lerp(27 - lifestyle_s["PHQ9_median"], 0, 27)  # invert: lower PHQ = higher score
    life_sleep = lerp(lifestyle_s["Sleep_hours_median"], 4, 8)
    lifestyle_avg = 0.6 * life_phq + 0.4 * life_sleep

    return {
        "risk_avg": round(risk_avg, 1),
        "reserve_avg": round(reserve_avg, 1),
        "lifestyle_avg": round(lifestyle_avg, 1),
        "components": {
            "risk": risk_components_scored,
            "reserve": res,
            "lifestyle": {"PHQ9_inverted": life_phq, "sleep_hours_scaled": life_sleep},
        }
    }


def main():
    print("[1/3] Risk components (NHANES 2017-2020)...")
    risk_s, _ = risk_components()
    print(f"  n_adults = {risk_s['n_adults']}, median age = {risk_s['median_age']}")
    for k,v in risk_s.items(): print(f"    {k}: {v}")

    print("\n[2/3] Reserve components (NHANES 2003-2014)...")
    reserve_s, *_ = reserve_components()
    for k,v in reserve_s.items(): print(f"    {k}: {v}")

    print("\n[3/3] Lifestyle components (NHANES 2017-2020)...")
    lifestyle_s, *_ = lifestyle_components()
    for k,v in lifestyle_s.items(): print(f"    {k}: {v}")

    print("\n[composite] Avg American on 0-100 axes:")
    avg = compute_avg_american(risk_s, reserve_s, lifestyle_s)
    print(json.dumps(avg, indent=2))

    out = {
        "raw": {"risk": risk_s, "reserve": reserve_s, "lifestyle": lifestyle_s},
        "avg_american": avg,
        "method": "NHANES public microdata; see calibration_inputs.md for component sources and anchor cutpoints",
    }
    with open(OUT/"avg_american.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT/'avg_american.json'}")


if __name__ == "__main__":
    main()
