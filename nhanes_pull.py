"""
NHANES public data pull for QoL calibration v2.

Downloads relevant NHANES cycles and computes the median values needed
to anchor the Avg American position on each axis.

Data source: CDC NHANES public XPT files.
Cycles used:
  2017-2020 (P): biomarkers, BP, BMI, smoking, A1c, demographics
  2011-2014 (G/H): grip strength (MGX), DEXA (DXX, DXXAG)
  1999-2002 (A/B): walking/balance (BAQ), VO2max submax (CVX)
  2011-2014 (G/H): cognitive (CFQ - DSST)
  2005-2018: PHQ-9 (DPQ)
  Various: PSQI not in NHANES (use BRFSS surrogate)

This script downloads what it can; missing surveys fall back to
published-literature defaults documented in calibration_inputs.md.
"""
import os
import sys
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "nhanes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# (filename, url) — NHANES URLs follow https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<YEAR>/DataFiles/<FILE>.xpt
# 2017-2020 was a combined pre-pandemic cycle ("P" suffix)
DATASETS = {
    # 2017-2020 pre-pandemic
    "DEMO_P":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt",
    "BPXO_P":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.xpt",  # BP (oscillometric)
    "BMX_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.xpt",   # body measures
    "TCHOL_P": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TCHOL.xpt", # total chol
    "HDL_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_HDL.xpt",   # HDL
    "TRIGLY_P":"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TRIGLY.xpt",# trig + LDL
    "GHB_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GHB.xpt",   # A1c
    "SMQ_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQ.xpt",   # smoking
    "DIQ_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.xpt",   # diabetes
    "BPQ_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.xpt",   # BP+chol meds
    "DPQ_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DPQ.xpt",   # PHQ-9
    "SLQ_P":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SLQ.xpt",   # sleep
    "BIOPRO_P":"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BIOPRO.xpt",# creatinine
    # 2013-2014 grip strength
    "MGX_H":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/MGX_H.xpt",
    "DEMO_H":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
    # 2011-2012 grip + DSST
    "MGX_G":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/MGX_G.xpt",
    "DEMO_G":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
    "CFQ_G":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/CFQ_G.xpt",
    # 2003-2004 VO2max submax
    "CVX_C":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/CVX_C.xpt",
    "DEMO_C":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/DEMO_C.xpt",
    # 1999-2000 VO2max
    "CVX":     "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/CVX.xpt",
    "DEMO":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DEMO.xpt",
}


def download_all():
    for name, url in DATASETS.items():
        out = DATA_DIR / f"{name}.xpt"
        if out.exists() and out.stat().st_size > 1000:
            continue
        print(f"[download] {name}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())
        except urllib.error.HTTPError as e:
            print(f"  ! {name}: HTTP {e.code}")
        except Exception as e:
            print(f"  ! {name}: {e}")


def read_xpt(name):
    import pyreadstat
    p = DATA_DIR / f"{name}.xpt"
    if not p.exists() or p.stat().st_size < 1000:
        return None
    df, _ = pyreadstat.read_xport(str(p))
    return df


if __name__ == "__main__":
    download_all()
    print("\n[manifest]")
    for name in DATASETS:
        p = DATA_DIR / f"{name}.xpt"
        size = p.stat().st_size if p.exists() else 0
        print(f"  {name:12s} {size:>10,} bytes")
