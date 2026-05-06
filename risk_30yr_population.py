"""
Re-score the 45-60 NHANES cohort with PREVENT 30-year and recompute
slider position using the 30-year distribution.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt
from prevent_30yr import prevent_total_cvd_30yr
from risk_population import build_individual_dataset

OUT = Path(__file__).parent / "calibration_outputs"


def main():
    df = build_individual_dataset()
    # 30-year is valid for ages 30-59
    df = df[(df["RIDAGEYR"] >= 30) & (df["RIDAGEYR"] <= 59)].copy()
    print(f"NHANES adults 30-59 with complete data: n={len(df)}")

    # Score every adult with PREVENT 30yr
    risks = []
    for _, row in df.iterrows():
        try:
            p = prevent_total_cvd_30yr(
                age=row["RIDAGEYR"], sex=row["sex"],
                total_chol=row["LBXTC"], hdl=row["LBDHDD"],
                sbp=row["SBP"], bp_treated=row["bp_treated"],
                diabetes=row["diabetes"], current_smoker=row["smoker"],
                egfr=row["eGFR"], bmi=row["BMXBMI"],
                statin=row["statin"],
            )
            risks.append(p)
        except Exception:
            risks.append(np.nan)
    df["prevent_30yr"] = risks
    df = df.dropna(subset=["prevent_30yr"])

    p_all = df["prevent_30yr"]
    print(f"\n30-year total CVD distribution (NHANES adults 30-59, n={len(df)}):")
    print(f"  mean   = {p_all.mean()*100:.1f}%")
    print(f"  median = {p_all.median()*100:.1f}%")
    print(f"  p1     = {p_all.quantile(0.01)*100:.1f}%")
    print(f"  p25    = {p_all.quantile(0.25)*100:.1f}%")
    print(f"  p75    = {p_all.quantile(0.75)*100:.1f}%")
    print(f"  p95    = {p_all.quantile(0.95)*100:.1f}%")
    print(f"  p99    = {p_all.quantile(0.99)*100:.1f}%")

    p1, p99 = p_all.quantile(0.01), p_all.quantile(0.99)

    # 45-59 cohort (capped at 59 due to 30-yr validity)
    cohort = df[(df["RIDAGEYR"]>=45) & (df["RIDAGEYR"]<=59)]
    p = cohort["prevent_30yr"]
    print(f"\n=== 45-59 cohort, 30-year CVD ===")
    print(f"  n = {len(cohort)}")
    print(f"  mean   = {p.mean()*100:.1f}%")
    print(f"  median = {p.median()*100:.1f}%")
    print(f"  p25    = {p.quantile(0.25)*100:.1f}%")
    print(f"  p75    = {p.quantile(0.75)*100:.1f}%")

    # Slider value using full-population anchors
    avg_slider = (p.mean() - p1) / (p99 - p1) * 100
    median_slider = (p.median() - p1) / (p99 - p1) * 100
    print(f"\n  On slider (anchored to full-cohort p1={p1*100:.1f}%, p99={p99*100:.1f}%):")
    print(f"    Avg 45-59 (mean)   = {avg_slider:.1f}")
    print(f"    Avg 45-59 (median) = {median_slider:.1f}")

    # Compare 10yr vs 30yr framing for same cohort
    print(f"\n=== Comparison for 45-59 cohort ===")
    print(f"  10-year framing: avg slider = 15.8 (population mean ~6%)")
    print(f"  30-year framing: avg slider = {avg_slider:.1f} (population mean {p.mean()*100:.0f}%)")

    # All-adult slider for reference
    all_avg_slider = (p_all.mean() - p1) / (p99 - p1) * 100
    print(f"  All adults 30-59: avg slider = {all_avg_slider:.1f} (mean {p_all.mean()*100:.0f}%)")

    # Sub-strata
    print(f"\n  Sub-strata (30-year):")
    for lo, hi in [(45,49),(50,54),(55,59)]:
        sub = cohort[(cohort["RIDAGEYR"]>=lo) & (cohort["RIDAGEYR"]<=hi)]
        sp = sub["prevent_30yr"]
        slider = (sp.mean() - p1) / (p99 - p1) * 100
        print(f"    {lo}-{hi}: n={len(sub)}, mean={sp.mean()*100:.1f}%, slider={slider:.1f}")

    out = {
        "method": "PREVENT 30-year, ratio-based extension of PREVENT 10-year",
        "n_full_cohort": int(len(df)),
        "full_cohort_distribution": {
            "mean_pct": float(p_all.mean()*100),
            "median_pct": float(p_all.median()*100),
            "p1_pct": float(p1*100),
            "p99_pct": float(p99*100),
        },
        "cohort_45_59": {
            "n": int(len(cohort)),
            "mean_pct": float(p.mean()*100),
            "median_pct": float(p.median()*100),
            "slider_mean": round(float(avg_slider), 1),
            "slider_median": round(float(median_slider), 1),
        },
        "all_adults_30_59_slider": round(float(all_avg_slider), 1),
    }
    with open(OUT/"risk_30yr.json","w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
