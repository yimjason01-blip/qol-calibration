"""
Re-cut the avg American calibration for ages 45-60 only — Meridian's
actual prevention-curious target demographic.

Two questions:
  1. What does the 45-60 cohort look like on Risk (PREVENT) vs all adults?
  2. What does it look like on Lifestyle (PHQ-9 + sleep) vs all adults?
  3. Does the Lifestyle composite as currently constructed accurately
     reflect midlife felt vitality, or is it missing the components
     that actually drag this cohort down?
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt
from prevent import prevent_total_cvd_10yr
from risk_population import build_individual_dataset, compute_prevent_for_all

OUT = Path(__file__).parent / "calibration_outputs"


def lerp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


def risk_45_60():
    df = build_individual_dataset()
    df = compute_prevent_for_all(df).dropna(subset=["prevent_10yr"])
    # Hold the global anchors (so the slider position is comparable across cohorts)
    p_all = df["prevent_10yr"]
    p1_global, p99_global = p_all.quantile(0.01), p_all.quantile(0.99)

    cohort = df[(df["RIDAGEYR"] >= 45) & (df["RIDAGEYR"] <= 60)]
    p = cohort["prevent_10yr"]
    print(f"=== RISK · 45-60 cohort ===")
    print(f"  n = {len(cohort)}")
    print(f"  PREVENT 10-yr CVD:")
    print(f"    mean   = {p.mean()*100:.2f}%")
    print(f"    median = {p.median()*100:.2f}%")
    print(f"    p25    = {p.quantile(0.25)*100:.2f}%")
    print(f"    p75    = {p.quantile(0.75)*100:.2f}%")
    print(f"    p95    = {p.quantile(0.95)*100:.2f}%")
    # Map to slider using GLOBAL anchors (so 45-60 avg can be compared to all-adult avg)
    avg_global = (p.mean() - p1_global) / (p99_global - p1_global) * 100
    median_global = (p.median() - p1_global) / (p99_global - p1_global) * 100
    print(f"  On global slider (anchored to all-adult p1={p1_global*100:.2f}%, p99={p99_global*100:.2f}%):")
    print(f"    Avg 45-60 (mean)   = {avg_global:.1f}")
    print(f"    Avg 45-60 (median) = {median_global:.1f}")
    print(f"  vs all-adult mean = 21.7  (so 45-60 is {avg_global-21.7:+.1f} points higher)")

    # Sub-stratify
    print(f"\n  Sub-strata:")
    for lo, hi in [(45,49),(50,54),(55,60)]:
        sub = cohort[(cohort["RIDAGEYR"]>=lo) & (cohort["RIDAGEYR"]<=hi)]
        sp = sub["prevent_10yr"]
        slider = (sp.mean() - p1_global) / (p99_global - p1_global) * 100
        print(f"    {lo}-{hi}: n={len(sub)}, mean PREVENT={sp.mean()*100:.2f}%, slider={slider:.1f}")

    return {
        "n": int(len(cohort)),
        "mean_pct": float(p.mean()*100),
        "median_pct": float(p.median()*100),
        "slider_mean_global": round(float(avg_global), 1),
        "slider_median_global": round(float(median_global), 1),
    }


def lifestyle_45_60():
    demo = read_xpt("DEMO_P")
    dpq  = read_xpt("DPQ_P")
    slq  = read_xpt("SLQ_P")
    a = demo[(demo["RIDAGEYR"]>=45) & (demo["RIDAGEYR"]<=60)][["SEQN","RIDAGEYR","RIAGENDR"]]

    phq_cols = [f"DPQ0{i}0" for i in range(1,10)]
    d = a.merge(dpq[["SEQN"]+phq_cols], on="SEQN", how="inner")
    for c in phq_cols:
        d[c] = d[c].where(d[c].isin([0,1,2,3]), np.nan)
    d["PHQ9_total"] = d[phq_cols].sum(axis=1, skipna=False)
    d = d.dropna(subset=["PHQ9_total"])
    d["PHQ9_0_100"] = d["PHQ9_total"].apply(lambda x: lerp(27 - x, 0, 27))

    s = a.merge(slq[["SEQN","SLD012"]], on="SEQN", how="inner").dropna(subset=["SLD012"])
    s["sleep_0_100"] = s["SLD012"].apply(lambda x: lerp(x, 4, 8))

    print(f"\n=== LIFESTYLE · 45-60 cohort (PHQ-9 + sleep only) ===")
    print(f"  PHQ-9: n={len(d)}, raw median={d['PHQ9_total'].median():.1f}, 0-100 median={d['PHQ9_0_100'].median():.1f}")
    print(f"  Sleep: n={len(s)}, raw median={s['SLD012'].median():.1f} hrs, 0-100 median={s['sleep_0_100'].median():.1f}")
    print(f"  PHQ-9 prevalence in 45-60:")
    print(f"    PHQ-9 ≥ 5  (any depression):    {(d['PHQ9_total']>=5).mean()*100:.1f}%")
    print(f"    PHQ-9 ≥ 10 (mod or severe):     {(d['PHQ9_total']>=10).mean()*100:.1f}%")
    print(f"  Sleep < 7 hrs:                    {(s['SLD012']<7).mean()*100:.1f}%")

    composite_pre = 0.6*d["PHQ9_0_100"].median() + 0.4*s["sleep_0_100"].median()
    composite_post = composite_pre - 8.0
    print(f"  Composite (60% PHQ + 40% sleep): {composite_pre:.1f} pre-adjustment, {composite_post:.1f} post -8 adjustment")
    return {
        "n_phq": int(len(d)),
        "n_sleep": int(len(s)),
        "PHQ9_median": float(d["PHQ9_total"].median()),
        "PHQ9_score": float(d["PHQ9_0_100"].median()),
        "sleep_median_hrs": float(s["SLD012"].median()),
        "sleep_score": float(s["sleep_0_100"].median()),
        "composite_pre_adj": round(composite_pre, 1),
        "composite_post_adj": round(composite_post, 1),
    }


def midlife_burden_evidence():
    """
    Pull additional NHANES tables that index dimensions the PHQ-9 + sleep
    composite misses, specifically for the 45-60 cohort.
    """
    print(f"\n=== ADDITIONAL MIDLIFE BURDEN INDICATORS (45-60) ===")
    demo = read_xpt("DEMO_P")
    a = demo[(demo["RIDAGEYR"]>=45) & (demo["RIDAGEYR"]<=60)][["SEQN","RIDAGEYR","RIAGENDR"]]

    # BMI / obesity prevalence (drags physical lifestyle / chronic-disease burden)
    bmx = read_xpt("BMX_P")
    b = a.merge(bmx[["SEQN","BMXBMI"]], on="SEQN", how="inner").dropna(subset=["BMXBMI"])
    print(f"  BMI:")
    print(f"    overweight or obese (≥25):    {(b['BMXBMI']>=25).mean()*100:.1f}%")
    print(f"    obese (≥30):                  {(b['BMXBMI']>=30).mean()*100:.1f}%")
    print(f"    severely obese (≥35):         {(b['BMXBMI']>=35).mean()*100:.1f}%")

    # Hypertension prevalence
    bpxo = read_xpt("BPXO_P")
    bp = a.merge(bpxo[["SEQN","BPXOSY1","BPXOSY2","BPXOSY3"]], on="SEQN", how="inner")
    bp["SBP"] = bp[["BPXOSY1","BPXOSY2","BPXOSY3"]].mean(axis=1, skipna=True)
    bp = bp.dropna(subset=["SBP"])
    print(f"  Hypertension:")
    print(f"    SBP ≥ 130:                    {(bp['SBP']>=130).mean()*100:.1f}%")
    print(f"    SBP ≥ 140:                    {(bp['SBP']>=140).mean()*100:.1f}%")

    # A1c
    ghb = read_xpt("GHB_P")
    g = a.merge(ghb[["SEQN","LBXGH"]], on="SEQN", how="inner").dropna(subset=["LBXGH"])
    print(f"  Glycemic:")
    print(f"    prediabetic (5.7 ≤ A1c < 6.5): {(((g['LBXGH']>=5.7)&(g['LBXGH']<6.5))).mean()*100:.1f}%")
    print(f"    diabetic (A1c ≥ 6.5):          {(g['LBXGH']>=6.5).mean()*100:.1f}%")

    # Smoking
    smq = read_xpt("SMQ_P")
    sm = a.merge(smq[["SEQN","SMQ040"]], on="SEQN", how="inner")
    sm["smoker"] = sm["SMQ040"].isin([1,2]).astype(int)
    print(f"  Current smoker:                  {sm['smoker'].mean()*100:.1f}%")

    # Self-reported general health (HSQ010)
    try:
        hsq = read_xpt("HSQ_P") if (Path(__file__).parent/"data/nhanes/HSQ_P.xpt").exists() else None
    except Exception:
        hsq = None
    if hsq is None:
        print(f"  Self-rated health: HSQ_P not pulled; would show fair/poor prevalence")


def main():
    risk_out = risk_45_60()
    life_out = lifestyle_45_60()
    midlife_burden_evidence()

    print("\n=== SUMMARY: 45-60 vs all-adult ===")
    print(f"  Risk:      45-60 = {risk_out['slider_mean_global']:.1f}  vs all-adult = 21.7   ({risk_out['slider_mean_global']-21.7:+.1f})")
    print(f"  Lifestyle: 45-60 = {life_out['composite_post_adj']:.1f}  vs all-adult = 82.6   ({life_out['composite_post_adj']-82.6:+.1f})")

    out = {"risk_45_60": risk_out, "lifestyle_45_60": life_out}
    with open(OUT/"avg_american_45_60.json","w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
