"""
Population-mean PREVENT: compute 10-year total CVD risk for every adult in
NHANES 2017-2020 with complete data, then take the population mean as
"Avg American risk."

Then map onto the 0-100 slider scale using anchors:
  0  = bottom 1st percentile of population PREVENT
  100 = top 99th percentile

This gives us a real, distribution-derived position for Avg American on
the Risk axis, instead of an averaged-mapping-of-medians fudge.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt
from prevent import prevent_total_cvd_10yr

OUT = Path(__file__).parent / "calibration_outputs"
OUT.mkdir(exist_ok=True)


def build_individual_dataset():
    demo = read_xpt("DEMO_P")
    bpxo = read_xpt("BPXO_P")
    bmx  = read_xpt("BMX_P")
    tchol = read_xpt("TCHOL_P")
    hdl  = read_xpt("HDL_P")
    ghb  = read_xpt("GHB_P")
    smq  = read_xpt("SMQ_P")
    diq  = read_xpt("DIQ_P")
    bpq  = read_xpt("BPQ_P")
    biopro = read_xpt("BIOPRO_P")

    df = demo[["SEQN","RIDAGEYR","RIAGENDR"]].copy()
    df = df[(df["RIDAGEYR"] >= 30) & (df["RIDAGEYR"] <= 79)]
    df = df.merge(bpxo[["SEQN","BPXOSY1","BPXOSY2","BPXOSY3"]], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN","BMXBMI"]], on="SEQN", how="left")
    df = df.merge(tchol[["SEQN","LBXTC"]], on="SEQN", how="left")
    df = df.merge(hdl[["SEQN","LBDHDD"]], on="SEQN", how="left")
    df = df.merge(ghb[["SEQN","LBXGH"]], on="SEQN", how="left")
    df = df.merge(diq[["SEQN","DIQ010","DIQ070"]], on="SEQN", how="left")  # DIQ070 = insulin/oral hypoglycemic
    df = df.merge(smq[["SEQN","SMQ020","SMQ040"]], on="SEQN", how="left")
    df = df.merge(biopro[["SEQN","LBXSCR"]], on="SEQN", how="left")
    # BP meds
    bpq_cols = [c for c in ["SEQN","BPQ050A","BPQ100D","BPQ090D"] if c in bpq.columns]
    df = df.merge(bpq[bpq_cols], on="SEQN", how="left")

    df["SBP"] = df[["BPXOSY1","BPXOSY2","BPXOSY3"]].mean(axis=1, skipna=True)
    df["smoker"] = df["SMQ040"].isin([1,2]).astype(bool)
    df["diabetes"] = (df["DIQ010"] == 1) | (df["LBXGH"] >= 6.5)
    df["bp_treated"] = df.get("BPQ050A", pd.Series([np.nan]*len(df))).isin([1]).astype(bool)
    df["statin"] = df.get("BPQ100D", pd.Series([np.nan]*len(df))).isin([1]).astype(bool)

    # eGFR (CKD-EPI 2021 race-free)
    def egfr(row):
        scr = row["LBXSCR"]
        if pd.isna(scr): return np.nan
        age = row["RIDAGEYR"]
        female = row["RIAGENDR"] == 2
        k = 0.7 if female else 0.9
        alpha = -0.241 if female else -0.302
        sex_mult = 1.012 if female else 1.0
        ratio = scr/k
        return 142 * (min(ratio,1)**alpha) * (max(ratio,1)**-1.200) * (0.9938**age) * sex_mult
    df["eGFR"] = df.apply(egfr, axis=1)

    df["sex"] = df["RIAGENDR"].map({1:"M", 2:"F"})

    # Need complete data
    needed = ["RIDAGEYR","sex","LBXTC","LBDHDD","SBP","BMXBMI","eGFR"]
    df = df.dropna(subset=needed).copy()
    return df


def compute_prevent_for_all(df):
    risks = []
    for _, row in df.iterrows():
        try:
            p = prevent_total_cvd_10yr(
                age=row["RIDAGEYR"],
                sex=row["sex"],
                total_chol=row["LBXTC"],
                hdl=row["LBDHDD"],
                sbp=row["SBP"],
                bp_treated=row["bp_treated"],
                diabetes=row["diabetes"],
                current_smoker=row["smoker"],
                egfr=row["eGFR"],
                bmi=row["BMXBMI"],
                statin=row["statin"],
            )
            risks.append(p)
        except Exception:
            risks.append(np.nan)
    df["prevent_10yr"] = risks
    return df


def main():
    print("[loading NHANES adults 30-79]...")
    df = build_individual_dataset()
    print(f"  n complete = {len(df):,}")
    print(f"  age median = {df['RIDAGEYR'].median():.0f}, range = [{df['RIDAGEYR'].min():.0f}, {df['RIDAGEYR'].max():.0f}]")

    print("[scoring with PREVENT]...")
    df = compute_prevent_for_all(df)
    df = df.dropna(subset=["prevent_10yr"])
    print(f"  n scored = {len(df):,}")

    p = df["prevent_10yr"]
    print(f"\n  10-year total CVD probability:")
    print(f"    mean   = {p.mean()*100:.2f}%")
    print(f"    median = {p.median()*100:.2f}%")
    print(f"    p5     = {p.quantile(0.05)*100:.2f}%")
    print(f"    p25    = {p.quantile(0.25)*100:.2f}%")
    print(f"    p75    = {p.quantile(0.75)*100:.2f}%")
    print(f"    p95    = {p.quantile(0.95)*100:.2f}%")
    print(f"    p99    = {p.quantile(0.99)*100:.2f}%")

    # Map mean adult to slider 0-100
    # Anchor: 0 = p1 of distribution, 100 = p99 of distribution
    p1, p99 = p.quantile(0.01), p.quantile(0.99)
    avg_pct = (p.mean() - p1) / (p99 - p1) * 100
    median_pct = (p.median() - p1) / (p99 - p1) * 100
    print(f"\n  On 0-100 risk slider (anchored to p1={p1*100:.2f}%, p99={p99*100:.2f}%):")
    print(f"    Avg American (mean)   = {avg_pct:.1f}")
    print(f"    Avg American (median) = {median_pct:.1f}")

    out = {
        "n": int(len(df)),
        "prevent_total_cvd_distribution": {
            "mean_pct": float(p.mean()*100),
            "median_pct": float(p.median()*100),
            "p1_pct": float(p1*100),
            "p5_pct": float(p.quantile(0.05)*100),
            "p25_pct": float(p.quantile(0.25)*100),
            "p75_pct": float(p.quantile(0.75)*100),
            "p95_pct": float(p.quantile(0.95)*100),
            "p99_pct": float(p99*100),
        },
        "slider_anchors": {
            "p1_anchor_for_0":  float(p1*100),
            "p99_anchor_for_100": float(p99*100),
        },
        "avg_american_slider_value": round(float(avg_pct), 1),
        "median_american_slider_value": round(float(median_pct), 1),
    }
    with open(OUT/"risk_avg_american.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT/'risk_avg_american.json'}")


if __name__ == "__main__":
    main()
