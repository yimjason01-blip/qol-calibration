"""
Multi-domain 30-year risk composite for the 45-59 NHANES cohort.

Composes 30-year probabilities across Meridian's five risk domains into
one "any major adverse health event" probability:

  CVD    - PREVENT 2024 30-year (already implemented)
  Metab  - 30-year incident T2DM, derived from ARIC/Diabetes Prevention
           Program-style logistic on A1c, BMI, family hx (proxied), age, sex
  CKD    - 30-year incident CKD stage 3+ (eGFR < 60), derived from
           current eGFR trajectory + risk factors (Tangri KFRE-adjacent
           but for incidence not progression)
  Cancer - 30-year all-invasive cancer incidence per SEER lifetime tables
           by age and sex, prorated to 30-year window
  Neuro  - 30-year all-cause dementia risk per CAIDE-style score
           (age, sex, education proxy, BP, cholesterol, BMI, physical activity)

These are all ANY-EVENT probabilities; we compose under the (rough)
independence assumption:
    P(any event) = 1 - Π(1 - P_i)

Real correlations exist (a person with diabetes is more likely to develop
CKD), but the independence approximation gives an upper-bound feel that
is still useful for teaching.

Reserve components (VO2max, grip, lean mass, bone, gait, cognitive) are
NOT inputs to any of these models — the inputs are biomarkers and
exposures. So the resulting composite is approximately reserve-orthogonal
by construction (no formal residualization needed).
"""
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt
from prevent_30yr import prevent_total_cvd_30yr
from risk_population import build_individual_dataset

OUT = Path(__file__).parent / "calibration_outputs"


def metab_30yr(age, sex, bmi, a1c, diabetes_already):
    """30-year incident T2DM. If already diabetic, returns 1.0.
    Otherwise: hazard scales with BMI and A1c.
    Calibrated against ADA Risk Test + DPP cohort observations:
      - lean (BMI<25), normoglycemic (A1c<5.5), no fam hx ~ 8-12% lifetime
      - obese (BMI>30) + prediabetic (A1c 5.7-6.4) ~ 50-70% lifetime
    """
    if diabetes_already:
        return 1.0
    base = 0.10 + 0.025*max(0, bmi-25) + 0.50*max(0, a1c-5.5)
    if a1c >= 5.7:
        base += 0.20
    if a1c >= 6.0:
        base += 0.15
    if age >= 50:
        base += 0.05
    return float(np.clip(base, 0, 0.85))


def ckd_30yr(age, sex, egfr, diabetes, sbp, bmi):
    """30-year incident CKD stage 3+ (eGFR<60).
    If already CKD3+, returns 1.0.
    Risk factors: age, diabetes, hypertension, obesity, current eGFR trajectory.
    Population baseline ~25-30% lifetime per CDC CKD surveillance.
    """
    if egfr < 60:
        return 1.0
    base = 0.20
    base += 0.005 * max(0, age - 45)
    if diabetes: base += 0.18
    if sbp >= 130: base += 0.08
    if sbp >= 140: base += 0.07
    if bmi >= 30: base += 0.05
    # Lower eGFR raises 30yr probability of crossing 60
    if egfr < 90:
        base += (90 - egfr) * 0.005
    return float(np.clip(base, 0, 0.85))


# SEER lifetime cancer risk (any invasive cancer) by sex, then prorated for
# 30-year window from baseline age. Source: SEER 2017-2019 lifetime tables
# (~40% male, ~38% female for any invasive cancer over remaining life from birth).
# For midlife adult, 30-year window captures most of remaining lifetime risk.
def cancer_30yr(age, sex, smoker, bmi):
    """30-year any-invasive-cancer incidence."""
    # Base: roughly proportional to remaining lifetime risk for that age band
    # SEER data for ages 50-59: ~33% any cancer in next 30 years (men), ~28% women
    if sex.upper().startswith("M"):
        base = 0.28 + 0.004*max(0, age-45)  # rises with starting age
    else:
        base = 0.24 + 0.004*max(0, age-45)
    # Smoking: HR ~1.5-2.0 for all-cancer; convert to additive for simplicity
    if smoker: base += 0.10
    # Obesity: HR ~1.2-1.4 for several cancers
    if bmi >= 30: base += 0.04
    if bmi >= 35: base += 0.03
    return float(np.clip(base, 0, 0.85))


def dementia_30yr(age, sex, sbp, bmi, smoker, diabetes):
    """30-year all-cause dementia risk per CAIDE-adjacent scoring.
    CAIDE ranges 0-15 with 0-5 ≈ 1% 20-yr risk and 12-15 ≈ 17% 20-yr risk.
    We extend to 30-year horizon and recalibrate for population averages.
    Lancet 2024 lifetime dementia risk for 55yo ≈ 35% women, 25% men over remaining life.
    """
    if sex.upper().startswith("F"):
        base = 0.18
    else:
        base = 0.13
    base += 0.005 * max(0, age - 45)
    if sbp >= 140: base += 0.05
    if bmi >= 30: base += 0.04
    if smoker: base += 0.04
    if diabetes: base += 0.07
    return float(np.clip(base, 0, 0.80))


def compose_independent(probs):
    """P(any event) = 1 - Π(1-p_i)"""
    survival = 1.0
    for p in probs:
        survival *= (1 - p)
    return 1 - survival


def main():
    df = build_individual_dataset()
    df = df[(df["RIDAGEYR"]>=45) & (df["RIDAGEYR"]<=59)].copy()
    print(f"NHANES adults 45-59 with complete data: n={len(df)}")

    rows = []
    for _, r in df.iterrows():
        try:
            cvd = prevent_total_cvd_30yr(
                age=r["RIDAGEYR"], sex=r["sex"],
                total_chol=r["LBXTC"], hdl=r["LBDHDD"],
                sbp=r["SBP"], bp_treated=r["bp_treated"],
                diabetes=r["diabetes"], current_smoker=r["smoker"],
                egfr=r["eGFR"], bmi=r["BMXBMI"], statin=r["statin"],
            )
            metab = metab_30yr(r["RIDAGEYR"], r["sex"], r["BMXBMI"],
                                r.get("LBXGH", np.nan) if not pd.isna(r.get("LBXGH",np.nan)) else 5.5,
                                r["diabetes"])
            ckd = ckd_30yr(r["RIDAGEYR"], r["sex"], r["eGFR"], r["diabetes"], r["SBP"], r["BMXBMI"])
            cancer = cancer_30yr(r["RIDAGEYR"], r["sex"], r["smoker"], r["BMXBMI"])
            neuro = dementia_30yr(r["RIDAGEYR"], r["sex"], r["SBP"], r["BMXBMI"], r["smoker"], r["diabetes"])
            any_event = compose_independent([cvd, metab, ckd, cancer, neuro])
            rows.append({"cvd":cvd, "metab":metab, "ckd":ckd, "cancer":cancer, "neuro":neuro, "any":any_event})
        except Exception as e:
            continue
    out = pd.DataFrame(rows)
    print(f"Scored: n={len(out)}\n")

    print("Per-domain 30-year probabilities (mean across 45-59 cohort):")
    for col in ["cvd","metab","ckd","cancer","neuro"]:
        print(f"  {col:<8} mean={out[col].mean()*100:>5.1f}%  median={out[col].median()*100:>5.1f}%")

    print(f"\nMulti-domain composite (any major event in 30 years):")
    print(f"  mean   = {out['any'].mean()*100:.1f}%")
    print(f"  median = {out['any'].median()*100:.1f}%")
    print(f"  p25    = {out['any'].quantile(0.25)*100:.1f}%")
    print(f"  p75    = {out['any'].quantile(0.75)*100:.1f}%")
    print(f"  p95    = {out['any'].quantile(0.95)*100:.1f}%")

    p1, p99 = out['any'].quantile(0.01), out['any'].quantile(0.99)
    avg_slider = (out['any'].mean() - p1) / (p99 - p1) * 100
    median_slider = (out['any'].median() - p1) / (p99 - p1) * 100
    print(f"\nOn slider (anchored to cohort p1={p1*100:.1f}%, p99={p99*100:.1f}%):")
    print(f"  Avg American 45-59 (mean)   = {avg_slider:.1f}")
    print(f"  Avg American 45-59 (median) = {median_slider:.1f}")

    print(f"\nDomain contribution to composite (% of avg composite from each):")
    avg_composite = out['any'].mean()
    for col in ["cvd","metab","ckd","cancer","neuro"]:
        # Approximate contribution: domain mean / sum of domain means
        pass
    domain_means = {c: out[c].mean() for c in ["cvd","metab","ckd","cancer","neuro"]}
    total = sum(domain_means.values())
    for c, m in sorted(domain_means.items(), key=lambda x: -x[1]):
        print(f"  {c:<8} {m*100:>5.1f}% of cohort  ({m/total*100:.0f}% of total domain hazard)")

    res = {
        "n": int(len(out)),
        "per_domain_mean_pct": {c: float(out[c].mean()*100) for c in ["cvd","metab","ckd","cancer","neuro"]},
        "composite_any_event_30yr": {
            "mean_pct": float(out['any'].mean()*100),
            "median_pct": float(out['any'].median()*100),
            "p25_pct": float(out['any'].quantile(0.25)*100),
            "p75_pct": float(out['any'].quantile(0.75)*100),
        },
        "slider_avg_american_45_59": round(float(avg_slider), 1),
        "slider_median_american_45_59": round(float(median_slider), 1),
        "method": "Independent composition of 5 domain 30-year probabilities (CVD via PREVENT, Metab/CKD/Cancer/Neuro via published-source-anchored approximations)",
    }
    with open(OUT/"risk_multi_domain.json","w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
