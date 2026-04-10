"""Local MDD evaluation: predictions CSV vs released test ground truth.

Truth file: data/raw/test_gt/ground_truth.csv (from `python -m src.data.download --datasets test_gt`)
Pred file:  CSV with columns ID, Prediction (space-separated phonemes)

Example:
  conda run -n iqraeval python scripts/evaluate_predictions.py \\
    --truth data/raw/test_gt/ground_truth.csv \\
    --pred outputs/predictions.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_from_csv, print_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", default="data/raw/test_gt/ground_truth.csv", help="ground_truth.csv from IqraEval_Test_GT")
    p.add_argument("--pred", required=True, help="Predictions CSV with ID, Prediction")
    args = p.parse_args()

    m = evaluate_from_csv(args.truth, args.pred)
    print_metrics(m, "Local MDD (open test + released GT)")
    print(
        "\nNote: Numbers should be comparable to the shared-task protocol if IDs align "
        "with open_testset. Small differences vs the old leaderboard can still happen "
        "(e.g. tokenizer whitespace, dataset version)."
    )


if __name__ == "__main__":
    main()
