import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction distributions and confusion matrices")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()

    paths = sorted((ROOT / "results" / args.split).glob("preds_*.csv"))
    if not paths:
        raise SystemExit(f"No prediction files in results/{args.split}")

    gold = pd.read_csv(paths[0])["label"].value_counts(normalize=True).round(3)
    print(f"Gold label distribution: {gold.to_dict()}")
    for path in paths:
        preds = pd.read_csv(path)
        distribution = preds["prediction"].value_counts(normalize=True).round(3)
        print(f"\n== {path.stem} ==\nPredicted: {distribution.to_dict()}")
        print(pd.crosstab(preds["label"], preds["prediction"], rownames=["gold"], colnames=["pred"]))


if __name__ == "__main__":
    main()
