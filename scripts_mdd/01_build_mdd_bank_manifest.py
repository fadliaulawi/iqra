#!/usr/bin/env python3
"""Build bank/train/dev manifests for retrieval-based MDD from unified metadata.

Scheme:
  - data/unified/train/ : utterance-level random split from all rows (correct+incorrect)
  - data/unified/dev/   : remaining utterances from all rows (correct+incorrect)
  - data/unified/bank/  : correct-only subset of train (memory bank candidates)

Each directory gets:
  - a manifest CSV
  - a metadata JSON with row/source/label stats
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "unified" / "metadata.csv"
DEFAULT_BANK_OUTPUT = REPO_ROOT / "data" / "unified" / "bank" / "manifest.csv"
DEFAULT_BANK_META = REPO_ROOT / "data" / "unified" / "bank" / "metadata.json"
DEFAULT_TRAIN_OUTPUT = REPO_ROOT / "data" / "unified" / "train" / "manifest.csv"
DEFAULT_TRAIN_META = REPO_ROOT / "data" / "unified" / "train" / "metadata.json"
DEFAULT_DEV_OUTPUT = REPO_ROOT / "data" / "unified" / "dev" / "manifest.csv"
DEFAULT_DEV_META = REPO_ROOT / "data" / "unified" / "dev" / "metadata.json"
DEFAULT_GLOBAL_STATS = REPO_ROOT / "data" / "unified" / "mdd_manifest_split_stats.json"


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _norm_phone(s: str) -> str:
    return " ".join(tok for tok in str(s).strip().split() if tok and tok != "<sil>")


def _finalize_manifest(df_in: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    keep_cols = [c for c in keep_cols if c in df_in.columns]
    return df_in[keep_cols].sort_values(["source", "orig_split", "ID"]).reset_index(drop=True)


def _phone_stats(out_df: pd.DataFrame) -> dict[str, object]:
    phone_counter = Counter()
    for seq in out_df.get("target_phoneme_sequence", pd.Series(dtype=str)).tolist():
        for tok in str(seq).split():
            phone_counter[tok] += 1
    by_source = out_df["source"].value_counts(dropna=False).to_dict() if "source" in out_df.columns else {}
    by_split = out_df["orig_split"].value_counts(dropna=False).to_dict() if "orig_split" in out_df.columns else {}
    return {
        "num_rows": int(len(out_df)),
        "unique_phones": int(len(phone_counter)),
        "top_phones": phone_counter.most_common(20),
        "by_source": by_source,
        "by_split": by_split,
        "avg_target_phone_len": float(out_df["num_target_phones"].mean()) if len(out_df) else 0.0,
    }


def _count_correct(df_in: pd.DataFrame) -> tuple[int, int]:
    if "is_correct" not in df_in.columns:
        return 0, 0
    c = int(df_in["is_correct"].map(_to_bool).sum())
    return c, int(len(df_in) - c)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create bank/train/dev MDD manifests from unified metadata.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--phone-col",
        default="phoneme_ref",
        choices=["phoneme_ref", "phonemes", "phoneme_aug"],
        help="Column to use as target/canonical sequence in manifests.",
    )
    parser.add_argument(
        "--strict-wav-check",
        action="store_true",
        help="Drop rows whose wav_path does not exist on disk.",
    )
    parser.add_argument(
        "--min-phone-count",
        type=int,
        default=1,
        help="Drop rows with fewer phoneme tokens than this.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of utterances assigned to train (rest to dev), from all rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for utterance split.")
    parser.add_argument(
        "--bank-output-csv",
        default=str(DEFAULT_BANK_OUTPUT),
        help="Bank manifest path (all correct).",
    )
    parser.add_argument(
        "--bank-metadata-json",
        default=str(DEFAULT_BANK_META),
        help="Per-dir metadata JSON for bank.",
    )
    parser.add_argument(
        "--train-output-csv",
        default=str(DEFAULT_TRAIN_OUTPUT),
        help="Train manifest path (mixed labels).",
    )
    parser.add_argument(
        "--train-metadata-json",
        default=str(DEFAULT_TRAIN_META),
        help="Per-dir metadata JSON for train.",
    )
    parser.add_argument("--dev-output-csv", default=str(DEFAULT_DEV_OUTPUT), help="Dev manifest path (mixed labels).")
    parser.add_argument(
        "--dev-metadata-json",
        default=str(DEFAULT_DEV_META),
        help="Per-dir metadata JSON for dev.",
    )
    parser.add_argument(
        "--global-stats-json",
        default=str(DEFAULT_GLOBAL_STATS),
        help="Global summary stats for all outputs.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input metadata not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = ["ID", "wav_path", "text", "source", "orig_split", "is_correct", args.phone_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Build canonical target sequence on full metadata first.
    df["target_phoneme_sequence"] = df[args.phone_col].map(_norm_phone)
    df["num_target_phones"] = df["target_phoneme_sequence"].map(lambda s: 0 if not s else len(s.split()))
    df = df[df["num_target_phones"] >= args.min_phone_count].copy()
    if df.empty:
        raise SystemExit("No rows left after min-phone-count filtering.")

    # Resolve wav paths and optionally check existence.
    wav_abs = []
    wav_exists = []
    for rel in tqdm(df["wav_path"].tolist(), desc="Resolving wav paths"):
        p = Path(str(rel))
        abs_p = p if p.is_absolute() else (REPO_ROOT / p)
        wav_abs.append(str(abs_p))
        wav_exists.append(abs_p.exists())
    df["wav_path_abs"] = wav_abs
    df["wav_exists"] = wav_exists

    if args.strict_wav_check:
        df = df[df["wav_exists"]].copy()
        if df.empty:
            raise SystemExit("No rows left after strict wav existence filtering.")

    keep_cols_common = [
        "ID",
        "wav_path",
        "wav_path_abs",
        "text",
        "target_phoneme_sequence",
        "num_target_phones",
        "source",
        "orig_split",
        "raw_label",
        "speaker",
        "sentence_ref",
        "sentence_spoken",
        "tashkeel_sentence",
        "duration_s",
    ]

    # Split all rows into train/dev by utterance ID (avoid leakage).
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise SystemExit("--train-fraction must be in (0, 1)")
    utt_ids = sorted(df["ID"].astype(str).unique().tolist())
    if not utt_ids:
        raise SystemExit("No utterances found after filtering.")
    rng = random.Random(int(args.seed))
    rng.shuffle(utt_ids)
    n_train_utts = max(1, int(round(len(utt_ids) * float(args.train_fraction))))
    train_utts = set(utt_ids[:n_train_utts])
    train_df = df[df["ID"].astype(str).map(lambda u: u in train_utts)].copy()
    dev_df = df[~df["ID"].astype(str).map(lambda u: u in train_utts)].copy()
    bank_df = train_df[train_df["is_correct"].map(_to_bool)].copy()
    if bank_df.empty:
        raise SystemExit("No correct rows found in train split for bank.")

    # Train/dev keep labels for supervised/eval analyses.
    keep_cols_mixed = keep_cols_common + ["is_correct", "phonemes", "phoneme_ref", "phoneme_aug"]
    bank_out_df = _finalize_manifest(bank_df, keep_cols_common)
    train_out_df = _finalize_manifest(train_df, keep_cols_mixed)
    dev_out_df = _finalize_manifest(dev_df, keep_cols_mixed)

    bank_out = Path(args.bank_output_csv)
    bank_meta = Path(args.bank_metadata_json)
    train_out = Path(args.train_output_csv)
    train_meta = Path(args.train_metadata_json)
    dev_out = Path(args.dev_output_csv)
    dev_meta = Path(args.dev_metadata_json)
    global_stats_out = Path(args.global_stats_json)

    bank_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    dev_out.parent.mkdir(parents=True, exist_ok=True)
    bank_out_df.to_csv(bank_out, index=False)
    train_out_df.to_csv(train_out, index=False)
    dev_out_df.to_csv(dev_out, index=False)

    bc, bw = _count_correct(bank_out_df)
    tc, tw = _count_correct(train_out_df)
    dc, dw = _count_correct(dev_out_df)

    bank_stats = {
        "manifest_csv": str(bank_out),
        "metadata_json": str(bank_meta),
        "selection": "correct_only_subset_of_train_split",
        "num_correct": bc,
        "num_incorrect": bw,
    }
    bank_stats.update(_phone_stats(bank_out_df))

    train_stats = {
        "manifest_csv": str(train_out),
        "metadata_json": str(train_meta),
        "selection": "random_utterance_split_from_all_rows",
        "split_role": "train",
        "num_correct": tc,
        "num_incorrect": tw,
    }
    train_stats.update(_phone_stats(train_out_df))

    dev_stats = {
        "manifest_csv": str(dev_out),
        "metadata_json": str(dev_meta),
        "selection": "random_utterance_split_from_all_rows",
        "split_role": "dev",
        "num_correct": dc,
        "num_incorrect": dw,
    }
    dev_stats.update(_phone_stats(dev_out_df))

    split_stats = {
        "input_csv": str(input_csv),
        "bank_output_csv": str(bank_out),
        "train_output_csv": str(train_out),
        "dev_output_csv": str(dev_out),
        "train_fraction": float(args.train_fraction),
        "seed": int(args.seed),
        "num_unique_utterances_total": int(len(utt_ids)),
        "num_unique_utterances_train": int(len(train_utts)),
        "num_unique_utterances_dev": int(len(utt_ids) - len(train_utts)),
        "num_missing_wav": int((~df["wav_exists"]).sum()) if "wav_exists" in df.columns else 0,
        "strict_wav_check": bool(args.strict_wav_check),
        "phone_column_used": args.phone_col,
        "min_phone_count": int(args.min_phone_count),
        "bank": bank_stats,
        "train": train_stats,
        "dev": dev_stats,
    }
    bank_meta.parent.mkdir(parents=True, exist_ok=True)
    train_meta.parent.mkdir(parents=True, exist_ok=True)
    dev_meta.parent.mkdir(parents=True, exist_ok=True)
    global_stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(bank_meta, "w", encoding="utf-8") as f:
        json.dump(bank_stats, f, ensure_ascii=False, indent=2)
    with open(train_meta, "w", encoding="utf-8") as f:
        json.dump(train_stats, f, ensure_ascii=False, indent=2)
    with open(dev_meta, "w", encoding="utf-8") as f:
        json.dump(dev_stats, f, ensure_ascii=False, indent=2)
    with open(global_stats_out, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    print(f"Saved bank manifest:  {bank_out} ({len(bank_out_df)} rows)")
    print(f"Saved train manifest: {train_out} ({len(train_out_df)} rows)")
    print(f"Saved dev manifest:   {dev_out} ({len(dev_out_df)} rows)")
    print(f"Saved bank metadata:  {bank_meta}")
    print(f"Saved train metadata: {train_meta}")
    print(f"Saved dev metadata:   {dev_meta}")
    print(f"Saved global stats:   {global_stats_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

