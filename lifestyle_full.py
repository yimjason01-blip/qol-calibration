"""
Full 10-dimension Lifestyle composite for the 45-59 cohort, anchored to
SPAREQ Vitality dimensions minus physical capacity (which lives in Reserve).

Architecture:
  - Dimensions where NHANES has microdata: use empirical population mean
    of the 0-100 mapped score (PHQ-9 mood, sleep duration).
  - Dimensions where NHANES lacks microdata: derive a population mean
    from published prevalence statistics using:
        pop_mean_dim = (1 - prevalence) * GOOD + prevalence * POOR
    where GOOD = 80 (asymptomatic-but-not-thriving) and POOR = 30
    (moderate-to-severe burden).

Sources for prevalence (each cited inline):
  - Mood: NHANES PHQ-9 microdata (already have)
  - Anxiety: CDC NHIS 2022 — symptoms of anxiety in past 2 weeks
  - Sleep: NHANES microdata (already have)
  - Energy/Drive: NHIS persistent fatigue/exhaustion ~20% adults
  - Libido: midlife averaged across sexes ~35% (FSFI / IIEF literature)
  - Connection: Surgeon General 2023 — ~30% moderate+ loneliness
  - Purpose: Pew/Gallup midlife purpose decline ~30% report low meaning
  - Pain: CDC NHIS 2021 — ~24% chronic pain in 45-64
  - Fatigue: NHIS — ~20% persistent fatigue
  - Cognitive complaints: BRFSS — ~12% subjective cognitive decline 45+

Weights: equal across 10 dimensions for v1. v2 candidate weights would
come from regressing SF-36 Vitality (or PROMIS Global) on these dimensions
in a cohort that has them all measured (not NHANES — needs UK Biobank
or HRS).
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nhanes_pull import read_xpt

OUT = Path(__file__).parent / "calibration_outputs"


def lerp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


GOOD_SCORE = 80   # asymptomatic-but-not-thriving baseline
POOR_SCORE = 30   # moderate-to-severe burden floor


def prevalence_to_score(prevalence_moderate_plus):
    """Convert population prevalence of moderate+ burden to mean dimension score."""
    return (1 - prevalence_moderate_plus) * GOOD_SCORE + prevalence_moderate_plus * POOR_SCORE


def main():
    # ---- NHANES-measured dimensions (45-59) ----
    demo = read_xpt("DEMO_P")
    dpq = read_xpt("DPQ_P")
    slq = read_xpt("SLQ_P")

    a = demo[(demo["RIDAGEYR"]>=45) & (demo["RIDAGEYR"]<=59)][["SEQN","RIDAGEYR"]]

    # Mood (PHQ-9)
    phq_cols = [f"DPQ0{i}0" for i in range(1,10)]
    d = a.merge(dpq[["SEQN"]+phq_cols], on="SEQN", how="inner")
    for c in phq_cols:
        d[c] = d[c].where(d[c].isin([0,1,2,3]), np.nan)
    d["PHQ9_total"] = d[phq_cols].sum(axis=1, skipna=False)
    d = d.dropna(subset=["PHQ9_total"])
    d["mood_0_100"] = d["PHQ9_total"].apply(lambda x: lerp(27 - x, 0, 27))
    mood_mean = float(d["mood_0_100"].mean())

    # Sleep
    s = a.merge(slq[["SEQN","SLD012"]], on="SEQN", how="inner").dropna(subset=["SLD012"])
    s["sleep_0_100"] = s["SLD012"].apply(lambda x: lerp(x, 4, 8))
    sleep_mean = float(s["sleep_0_100"].mean())

    # ---- Prevalence-derived dimensions (45-59 estimates) ----
    # Each prevalence figure cited inline; midlife-specific where available.
    dimensions = {
        "mood":              {"score": mood_mean, "source": "NHANES PHQ-9 microdata, n="+str(len(d)), "method": "empirical mean"},
        "anxiety":           {"prevalence": 0.20, "source": "CDC NHIS 2022 - past-2-wk anxiety symptoms"},
        "sleep":             {"score": sleep_mean, "source": "NHANES sleep duration, n="+str(len(s)), "method": "empirical mean"},
        "energy_drive":      {"prevalence": 0.20, "source": "NHIS persistent low energy ~20%"},
        "libido":            {"prevalence": 0.35, "source": "FSFI/IIEF midlife literature, sex-averaged"},
        "connection":        {"prevalence": 0.30, "source": "Surgeon General 2023 loneliness moderate+"},
        "purpose":           {"prevalence": 0.30, "source": "Gallup/Pew low-meaning midlife"},
        "pain":              {"prevalence": 0.24, "source": "CDC NHIS 2021 chronic pain age 45-64"},
        "fatigue":           {"prevalence": 0.20, "source": "NHIS persistent fatigue"},
        "cognitive_complaint":{"prevalence": 0.12, "source": "BRFSS SCD adults 45+"},
    }

    # Compute score for each
    print("=== 10-dimension Lifestyle composite (45-59 cohort) ===\n")
    print(f"{'Dimension':<22} {'Score':>6}  {'Method':<40} {'Source'}")
    scores = []
    for name, d_info in dimensions.items():
        if "score" in d_info:
            score = d_info["score"]
            method = d_info["method"]
        else:
            score = prevalence_to_score(d_info["prevalence"])
            method = f"{d_info['prevalence']*100:.0f}% moderate+ → score"
            d_info["score"] = score
        scores.append(score)
        print(f"  {name:<20} {score:>6.1f}  {method:<40} {d_info['source']}")

    # Equal weights, v1
    composite = float(np.mean(scores))
    print(f"\nEqual-weight composite: {composite:.1f}")

    # Compare to v2 (PHQ + sleep only)
    print(f"\nv2 (PHQ + sleep only, with -8 hand-wave):  82.6")
    print(f"v5 (10-dimension SPAREQ Vitality):         {composite:.1f}")
    print(f"Δ:                                         {composite-82.6:+.1f}")

    # Sanity check: where does Avg American land relative to slider?
    # Slider 0 = chronic poor on all 10, slider 100 = optimal on all 10
    # Composite already on 0-100 scale, so direct mapping
    print(f"\nAvg American 45-59 Lifestyle slider position: {composite:.0f}")

    out = {
        "method": "10-dimension SPAREQ Vitality composite (physical components removed; those live in Reserve)",
        "cohort": "NHANES 45-59",
        "anchors": {"GOOD_SCORE": GOOD_SCORE, "POOR_SCORE": POOR_SCORE},
        "dimensions": dimensions,
        "composite_avg_american_45_59": round(composite, 1),
        "previous_v2_value": 82.6,
        "delta": round(composite - 82.6, 1),
    }
    with open(OUT/"lifestyle_v5_full.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT/'lifestyle_v5_full.json'}")


if __name__ == "__main__":
    main()
