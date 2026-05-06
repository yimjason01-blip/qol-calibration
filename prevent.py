"""
AHA PREVENT 2024 — 10-year CVD risk equation, total CVD endpoint.

Source: Khan SS et al, Circulation 2024 (AHA PREVENT). Coefficients for
the base model (without optional kidney/social/glycemic enhancements).

Inputs:
  age (years, 30-79)
  sex ('M' or 'F')
  total_chol (mg/dL)
  hdl (mg/dL)
  sbp (mmHg)
  bp_treated (bool)
  diabetes (bool)
  current_smoker (bool)
  egfr (mL/min/1.73m2)
  bmi (kg/m^2)
  statin (bool)

Output: 10-year probability of total CVD (MI + stroke + CV death + HF), 0-1.

Note: PREVENT was developed in a contemporary multi-cohort and is the
current AHA-endorsed CVD risk model (replacing PCE 2013). Coefficients
below are for the "base" total CVD equation. Race-free by design.
"""
import math


def prevent_total_cvd_10yr(
    age: float,
    sex: str,            # 'M' or 'F'
    total_chol: float,
    hdl: float,
    sbp: float,
    bp_treated: bool,
    diabetes: bool,
    current_smoker: bool,
    egfr: float,
    bmi: float,
    statin: bool,
) -> float:
    # Centered/transformed inputs per Khan 2024 supplement Table S6 (Total CVD, base model)
    age_c       = (age - 55) / 10
    nonhdl      = total_chol - hdl
    nonhdl_c    = (nonhdl * 0.02586) - 3.5         # mmol/L; convert mg/dL → mmol/L by *0.02586
    hdl_c       = ((hdl * 0.02586) - 1.3) / 0.3
    sbp_lt_min  = (min(sbp, 110) - 110) / 20
    sbp_ge_min  = (max(sbp, 110) - 130) / 20
    egfr_lt     = (min(egfr, 60) - 60) / -15
    egfr_ge     = (max(egfr, 60) - 90) / -15
    bmi_lt30    = (min(bmi, 30) - 25) / 5
    bmi_ge30    = (max(bmi, 30) - 30) / 5

    if sex.upper().startswith("F"):
        # Female total CVD coefficients
        b = (
            0.7939329 * age_c
            + 0.0305239 * nonhdl_c
            + -0.1606857 * hdl_c
            + -0.2394003 * sbp_lt_min
            + 0.3600781 * sbp_ge_min
            + 0.8667604 * float(diabetes)
            + 0.5360739 * float(current_smoker)
            + 0.6045917 * egfr_lt
            + 0.0433769 * egfr_ge
            + 0.3151672 * float(bp_treated)
            + -0.1477655 * float(statin)
            + -0.0663612 * (sbp_ge_min * float(bp_treated))
            + 0.1197879 * (nonhdl_c * float(statin))
            + -0.0819715 * (age_c * nonhdl_c)
            + 0.0306769 * (age_c * hdl_c)
            + -0.0946348 * (age_c * sbp_ge_min)
            + -0.27057 * (age_c * float(diabetes))
            + -0.078715 * (age_c * float(current_smoker))
            + -0.1637806 * (age_c * egfr_lt)
        )
        const = -3.307728
    else:
        # Male total CVD coefficients
        b = (
            0.7688528 * age_c
            + 0.0736174 * nonhdl_c
            + -0.0954431 * hdl_c
            + -0.4347345 * sbp_lt_min
            + 0.3362658 * sbp_ge_min
            + 0.7692857 * float(diabetes)
            + 0.4386871 * float(current_smoker)
            + 0.5378979 * egfr_lt
            + 0.0164827 * egfr_ge
            + 0.288879 * float(bp_treated)
            + -0.1337349 * float(statin)
            + -0.0475924 * (sbp_ge_min * float(bp_treated))
            + 0.150273 * (nonhdl_c * float(statin))
            + -0.0517874 * (age_c * nonhdl_c)
            + 0.0191169 * (age_c * hdl_c)
            + -0.1049477 * (age_c * sbp_ge_min)
            + -0.2251948 * (age_c * float(diabetes))
            + -0.0895067 * (age_c * float(current_smoker))
            + -0.1543702 * (age_c * egfr_lt)
        )
        const = -3.031168

    # logit-link: P = 1 / (1 + exp(-(const + b)))
    logit = const + b
    p = 1.0 / (1.0 + math.exp(-logit))
    return p


if __name__ == "__main__":
    # Sanity check: avg American profile from NHANES medians
    p = prevent_total_cvd_10yr(
        age=50, sex="M",
        total_chol=183, hdl=51,
        sbp=121, bp_treated=False,
        diabetes=False, current_smoker=False,
        egfr=98, bmi=29, statin=False,
    )
    print(f"Median 50yo M, NHANES profile: 10-yr total CVD = {p*100:.1f}%")

    p = prevent_total_cvd_10yr(
        age=50, sex="F",
        total_chol=183, hdl=51,
        sbp=121, bp_treated=False,
        diabetes=False, current_smoker=False,
        egfr=98, bmi=29, statin=False,
    )
    print(f"Median 50yo F, NHANES profile: 10-yr total CVD = {p*100:.1f}%")

    # Stress test: high-risk profile
    p = prevent_total_cvd_10yr(
        age=65, sex="M",
        total_chol=240, hdl=35,
        sbp=160, bp_treated=True,
        diabetes=True, current_smoker=True,
        egfr=55, bmi=33, statin=False,
    )
    print(f"High-risk 65yo M: 10-yr total CVD = {p*100:.1f}%")

    # Optimal profile
    p = prevent_total_cvd_10yr(
        age=50, sex="M",
        total_chol=160, hdl=60,
        sbp=110, bp_treated=False,
        diabetes=False, current_smoker=False,
        egfr=100, bmi=23, statin=False,
    )
    print(f"Optimal 50yo M: 10-yr total CVD = {p*100:.1f}%")
