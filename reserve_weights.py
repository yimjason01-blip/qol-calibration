"""
Literature-anchored reserve weights.

Replaces the v1 clinical-prior weights with weights derived from published
mortality hazard ratios per standard-deviation increment. We take |log(HR)|
for each component (per SD), normalize to sum to 1.0, and use as weights.

This is not a UK Biobank Cox fit — that requires cohort access. But it
is a defensible mid-step that grounds weights in published mortality
effect sizes rather than clinical opinion.

Sources (all per-SD increments, all-cause mortality, age- and sex-adjusted
where available):

  VO2max        Kodama 2009 JAMA meta-analysis: HR 0.87 per 1-MET (~3.5 mL/kg/min)
                increment. Convert to per-SD: SD ~ 9 mL/kg/min ~ 2.6 METs
                → HR ~ 0.87^2.6 = 0.69 per SD
  Grip strength Leong 2015 Lancet (PURE 17-country): HR 0.84 per 5-kg
                increment. SD ~ 11 kg → HR 0.84^2.2 = 0.66 per SD
  ALMI (lean)   Cawthon 2015 et al, ALMI: HR ~0.75-0.85 per SD increment
                (sex-stratified). Use HR 0.80 per SD.
  Bone density  Johnell 2005 Osteoporos Int: HR 1.5 per SD DECREASE in BMD
                → HR 0.67 per SD INCREASE.
  Gait speed    Studenski 2011 JAMA: HR 0.90 per 0.1 m/s increment.
                SD ~ 0.20 m/s → HR 0.90^2 = 0.81 per SD
  DSST          Lara 2016 et al cognitive reserve: HR ~0.83 per SD
                (processing speed composite mortality)
  Lower body
   power        Skelton 1995 / Reid 2012: per-SD HR ~0.78 for leg power
  Pulmonary
   FEV1         Schunemann 2000: HR 0.83 per SD FEV1 % predicted

Weights below are proportional to |log(HR)|.
"""
import math
import json
from pathlib import Path

OUT = Path(__file__).parent / "calibration_outputs"
OUT.mkdir(exist_ok=True)

components = {
    "VO2max":         {"hr_per_sd": 0.69, "source": "Kodama 2009 JAMA"},
    "grip":           {"hr_per_sd": 0.66, "source": "Leong 2015 Lancet (PURE)"},
    "ALMI":           {"hr_per_sd": 0.80, "source": "Cawthon 2015"},
    "bone_density":   {"hr_per_sd": 0.67, "source": "Johnell 2005 Osteoporos Int"},
    "gait_speed":     {"hr_per_sd": 0.81, "source": "Studenski 2011 JAMA"},
    "DSST_processing":{"hr_per_sd": 0.83, "source": "Lara 2016 / cognitive-reserve lit"},
    "leg_power":      {"hr_per_sd": 0.78, "source": "Reid 2012"},
    "FEV1":           {"hr_per_sd": 0.83, "source": "Schunemann 2000"},
}

# Compute |log(HR)| for each
for k, v in components.items():
    v["log_hr_abs"] = abs(math.log(v["hr_per_sd"]))

total = sum(v["log_hr_abs"] for v in components.values())
for k, v in components.items():
    v["weight"] = round(v["log_hr_abs"] / total, 4)

print("Literature-anchored reserve weights (mortality |log(HR)|-derived):")
print(f"{'Component':<20} {'HR/SD':>7} {'|log(HR)|':>11} {'Weight':>8}")
for k, v in sorted(components.items(), key=lambda x: -x[1]["weight"]):
    print(f"  {k:<18} {v['hr_per_sd']:>6.2f}  {v['log_hr_abs']:>10.4f}  {v['weight']*100:>6.1f}%")

print(f"\nSum of weights: {sum(v['weight'] for v in components.values()):.4f}")

print("\nCompare to v1 clinical-prior weights:")
v1 = {"VO2max": 0.30, "ALMI": 0.15, "grip": 0.10, "leg_power": 0.10, "bone_density": 0.10, "gait_speed": 0.10, "DSST_processing": 0.10, "FEV1": 0.05}
print(f"{'Component':<20} {'v1 (prior)':>10} {'v2 (literature)':>16}")
for k in v1:
    v1w = v1[k]
    v2w = components[k]["weight"]
    delta = v2w - v1w
    arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
    print(f"  {k:<18} {v1w*100:>8.0f}%  {v2w*100:>14.1f}%  {arrow} {delta*100:+.1f}")

with open(OUT/"reserve_weights_v2.json","w") as f:
    json.dump({"components": components, "v1_priors": v1}, f, indent=2)
print(f"\nWrote {OUT/'reserve_weights_v2.json'}")
