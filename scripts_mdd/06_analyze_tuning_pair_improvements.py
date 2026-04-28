#!/usr/bin/env python3
"""Analyze pair-level error reduction from rule tuning across k.

Compares base vs tuned retrieval stats and reports which
(expected_phoneme, majority_label) pairs were reduced the most.

Default analysis target is FR-only pair table:
  retrieval_crosstab.fr_rows_gold_ok_pred_wrong

Also supports:
  retrieval_crosstab.expected_vs_majority_all_query_rows
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_pair_counts(stats: dict[str, Any], table_key: str) -> dict[tuple[str, str], int]:
    crosstab = stats.get("retrieval_crosstab", {})
    rows = crosstab.get(table_key, [])
    out: dict[tuple[str, str], int] = {}
    for r in rows:
        e = str(r.get("expected_phoneme", ""))
        m = str(r.get("majority_label", ""))
        c = int(r.get("count", 0))
        if not e or not m:
            continue
        out[(e, m)] = c
    return out


def _pair_rows_for_diff(
    base_counts: dict[tuple[str, str], int],
    tuned_counts: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = set(base_counts) | set(tuned_counts)
    for key in keys:
        b = int(base_counts.get(key, 0))
        t = int(tuned_counts.get(key, 0))
        delta = b - t  # positive => error reduction
        pct = (delta / b * 100.0) if b > 0 else 0.0
        rows.append(
            {
                "expected_phoneme": key[0],
                "majority_label": key[1],
                "base_count": b,
                "tuned_count": t,
                "reduction_count": delta,
                "reduction_pct_of_base": round(pct, 3),
            }
        )
    rows.sort(key=lambda x: (x["reduction_count"], x["base_count"]), reverse=True)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "expected_phoneme",
                "majority_label",
                "base_count",
                "tuned_count",
                "reduction_count",
                "reduction_pct_of_base",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pair-level error reduction from tuned vote weights.")
    parser.add_argument(
        "--test-root",
        default="data/unified/test",
        help="Root containing retrieval_{k}/ folders.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
        help="k values to analyze.",
    )
    parser.add_argument(
        "--base-file",
        default="mdd_retrieval_base_stats.json",
        help="Filename for base stats inside retrieval_{k}/.",
    )
    parser.add_argument(
        "--tuned-file",
        default="mdd_retrieval_stats.json",
        help="Filename for tuned stats inside retrieval_{k}/.",
    )
    parser.add_argument(
        "--table-key",
        default="fr_rows_gold_ok_pred_wrong",
        choices=["fr_rows_gold_ok_pred_wrong", "expected_vs_majority_all_query_rows"],
        help="Pair table to compare.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Top reduced pairs to include in summary JSON.",
    )
    args = parser.parse_args()

    root = Path(args.test_root)
    overall_summary: dict[str, Any] = {
        "table_key": args.table_key,
        "results": {},
    }

    for k in args.ks:
        k_dir = root / f"retrieval_{k}"
        base_path = k_dir / args.base_file
        tuned_path = k_dir / args.tuned_file
        if not base_path.is_file() or not tuned_path.is_file():
            print(f"[skip] k={k}: missing file(s): {base_path.name} or {tuned_path.name}")
            continue

        base = _load_json(base_path)
        tuned = _load_json(tuned_path)
        base_counts = _extract_pair_counts(base, args.table_key)
        tuned_counts = _extract_pair_counts(tuned, args.table_key)
        rows = _pair_rows_for_diff(base_counts, tuned_counts)

        csv_out = k_dir / f"pair_reduction_{args.table_key}.csv"
        _write_csv(csv_out, rows)

        total_base = sum(base_counts.values())
        total_tuned = sum(tuned_counts.values())
        total_reduction = total_base - total_tuned
        total_reduction_pct = (total_reduction / total_base * 100.0) if total_base > 0 else 0.0

        best_rows = [r for r in rows if r["reduction_count"] > 0][: args.top_n]
        worst_rows = sorted(rows, key=lambda x: x["reduction_count"])[: args.top_n]

        summary = {
            "k": k,
            "base_stats_path": str(base_path),
            "tuned_stats_path": str(tuned_path),
            "csv_output": str(csv_out),
            "num_pairs_base": len(base_counts),
            "num_pairs_tuned": len(tuned_counts),
            "total_base_count": int(total_base),
            "total_tuned_count": int(total_tuned),
            "total_reduction_count": int(total_reduction),
            "total_reduction_pct_of_base": round(total_reduction_pct, 3),
            "top_reduced_pairs": best_rows,
            "top_increased_pairs": worst_rows,
        }
        overall_summary["results"][str(k)] = summary
        print(
            f"[k={k}] wrote {csv_out.name} | "
            f"base={total_base} tuned={total_tuned} reduction={total_reduction} "
            f"({total_reduction_pct:.2f}%)"
        )

    summary_out = root / f"pair_reduction_summary_{args.table_key}.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

