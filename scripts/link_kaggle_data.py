"""Find the dataset files anywhere under /kaggle/input and symlink them into data/ with the names gc-ids expects.

    python scripts/link_kaggle_data.py [--input /kaggle/input] [--data data]
Prints what was found / missing. Safe to re-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

WANT = {
    "nslkdd": ["KDDTrain+.txt", "KDDTest+.txt"],
    "unswnb15": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    root, data = Path(a.input), Path(a.data)
    for ds, files in WANT.items():
        (data / ds).mkdir(parents=True, exist_ok=True)
        for f in files:
            hits = [p for p in root.rglob("*") if p.name.lower() == f.lower()]
            dst = data / ds / f
            if hits:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(hits[0].resolve())
                print(f"OK      {ds}/{f} <- {hits[0]}")
            else:
                print(f"MISSING {ds}/{f}  (add the dataset to the notebook: Add Input)")
    cic = sorted({p.parent for p in root.rglob("*.csv") if "Wednesday-workingHours" in p.name or "Friday-WorkingHours" in p.name})
    (data / "cicids2017").mkdir(parents=True, exist_ok=True)
    if cic:
        dst = data / "cicids2017" / "MachineLearningCVE"
        if dst.exists() or dst.is_symlink():
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(cic[0].resolve())
        print(f"OK      cicids2017 <- {cic[0]}")
    else:
        print("INFO    CIC-IDS2017 not found (optional third dataset)")


if __name__ == "__main__":
    main()
