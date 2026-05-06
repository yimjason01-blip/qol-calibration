# QoL Calibration

Source code for the v2 calibration of the Quality of Life teaching tool.

**Live prototype:** https://yimjason01-blip.github.io/qol-three-slider/
**Calibration spec:** https://yimjason01-blip.github.io/qol-three-slider/spec.html

## What this repo does

Pulls NHANES public microdata, scores 5,597 adults with AHA PREVENT 2024,
computes population-anchored Avg American positions on each of the three
axes (Risk, Reserve, Lifestyle), and derives literature-anchored reserve
weights from published mortality hazard ratios.

## Files

- `nhanes_pull.py` — downloads required NHANES XPT files
- `prevent.py` — AHA PREVENT 2024 total CVD 10-year equation
- `risk_population.py` — scores all NHANES adults 30-79, derives Avg American risk
- `reserve_lifestyle.py` — Reserve (VO2max + grip + DSST) and Lifestyle (PHQ-9 + sleep) composites
- `composite_v2.py` — earlier draft, superseded
- `reserve_weights.py` — literature-anchored reserve weights from |log(HR)|
- `calibration_outputs/` — JSON outputs

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install pyreadstat pandas numpy scipy
python nhanes_pull.py        # downloads ~50MB
python risk_population.py
python reserve_lifestyle.py
python reserve_weights.py
```

## Avg American v2 (NHANES-derived, replacing eyeballed v1)

| Axis      | v1 (eyeballed) | v2 (calibrated) | Method |
|-----------|----------------|------------------|--------|
| Risk      | 65             | **22**           | PREVENT 2024 total CVD, n=5,597 |
| Reserve   | 28             | **50**           | VO2max + grip + DSST medians, NHANES |
| Lifestyle | 42             | **83**           | PHQ-9 + sleep, with -8 uncovered adjustment |

## Known gaps

- Risk is CVD-only. Cancer, dementia, fracture not included; these would shift Avg American higher.
- Reserve covers 3 of 8 spec components. DEXA, bone density, gait speed, lower-body power, FEV1 require either NHANES cycles we haven't pulled or non-NHANES sources.
- Lifestyle covers 2 of 10 spec components. Loneliness, pain, fatigue, libido, purpose, etc. require BRFSS / Surgeon General / external sources. The -8 adjustment is documented but approximate.
- Curve geometry and render mix are NOT yet calibrated. Those require UK Biobank longitudinal data.

## Status: 5/10 accuracy

Architecture: 8/10 (publication-quality, survives recalibration)
Avg American: 7/10 (real population data, with documented gaps)
Curve geometry: 2/10 (still hand-tuned)
Render mix: 2/10 (still 0.5/0.5 asserted)
