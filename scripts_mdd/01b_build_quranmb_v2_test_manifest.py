#!/usr/bin/env python3
"""Build a unified-style test manifest from two fixed Hugging Face datasets.

- Audio: IqraEval/QuranMB.v2 (test split)
- Gold labels: IqraEval/IqraEval_Test_GT (test) — Reference_phn, Annotation_phn (gated)

The layout matches 01_build_mdd_bank_manifest.py (train style) for 02_align_and_segment.py and
downstream MDD scripts (--dataset-set custom).

Authentication: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment and accept the gold
dataset terms on the Hub. Optional --merge-ref-csv overlays or replaces phoneme columns.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import wave
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "unified" / "test"
# Fixed Hub sources (not configurable).
HF_AUDIO = "IqraEval/QuranMB.v2"
HF_AUDIO_SPLIT = "test"
HF_GOLD = "IqraEval/IqraEval_Test_GT"
HF_GOLD_SPLIT = "test"

from dotenv import load_dotenv
load_dotenv()

def _apply_hf_token_from_env() -> None:
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")).strip()
    if t:
        os.environ["HF_TOKEN"] = t
        os.environ["HUGGING_FACE_HUB_TOKEN"] = t


def _rel_to_repo(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _norm_phone(s: str) -> str:
    return " ".join(tok for tok in str(s).strip().split() if tok and tok != "<sil>")


def _wav_duration_sec_from_bytes(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        n = w.getnframes()
        r = w.getframerate()
        if r <= 0:
            return 0.0
        return float(n) / float(r)


def _phone_stats(out_df: pd.DataFrame) -> dict[str, object]:
    phone_counter: Counter[str] = Counter()
    for seq in out_df.get("target_phoneme_sequence", pd.Series(dtype=str)).tolist():
        for tok in str(seq).split():
            if tok:
                phone_counter[tok] += 1
    by_source = out_df["source"].value_counts(dropna=False).to_dict() if "source" in out_df.columns else {}
    n_nonempty = int((out_df["num_target_phones"] > 0).sum()) if "num_target_phones" in out_df.columns else 0
    return {
        "num_rows": int(len(out_df)),
        "rows_with_nonempty_target_phonemes": n_nonempty,
        "unique_phones": int(len(phone_counter)),
        "top_phones": phone_counter.most_common(20),
        "by_source": by_source,
    }


def _merge_ref_phonemes(df: pd.DataFrame, ref_csv: Path, ref_col: str) -> pd.DataFrame:
    ref = pd.read_csv(ref_csv)
    if "ID" not in ref.columns:
        raise SystemExit(f"--merge-ref-csv must contain an 'ID' column. Got: {list(ref.columns)}")

    candidates = [ref_col, "phoneme_ref", "Reference_phn", "reference_phn", "target_phoneme_sequence"]
    use_col = next((c for c in candidates if c in ref.columns), None)
    if use_col is None:
        raise SystemExit(
            f"No reference-phoneme column in merge CSV. Tried {candidates!r}. Columns: {list(ref.columns)}"
        )

    m = ref[["ID", use_col]].drop_duplicates("ID", keep="first").rename(columns={use_col: "phoneme_ref_merged"})
    out = df.merge(m, on="ID", how="left")
    merged = out["phoneme_ref_merged"]
    out["phoneme_ref"] = merged.combine_first(out["phoneme_ref"])
    out["phoneme_ref"] = out["phoneme_ref"].fillna("").map(lambda s: str(s) if s is not None else "")
    out = out.drop(columns=["phoneme_ref_merged"], errors="ignore")

    for ann_name in ("Annotation_phn", "phonemes"):
        if ann_name in ref.columns:
            a = ref[["ID", ann_name]].drop_duplicates("ID", keep="first").rename(
                columns={ann_name: "ann_merged"}
            )
            out = out.merge(a, on="ID", how="left")
            out["phonemes"] = out["ann_merged"].combine_first(out["phonemes"])
            out = out.drop(columns=["ann_merged"], errors="ignore")
            out["phonemes"] = out["phonemes"].fillna("").map(lambda s: str(s) if s is not None else "")
            out["phoneme_aug"] = out["phonemes"]
            break

    out["target_phoneme_sequence"] = out["phoneme_ref"].map(_norm_phone)
    out["num_target_phones"] = out["target_phoneme_sequence"].map(
        lambda s: 0 if not s else len(str(s).split())
    )
    return out


def _apply_hf_gold(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise SystemExit("Install `datasets` (Hugging Face): pip install datasets") from e
    try:
        gt = load_dataset(HF_GOLD, split=HF_GOLD_SPLIT, trust_remote_code=True)
    except Exception as e:
        raise SystemExit(
            f"Failed to load gold {HF_GOLD!r} (split={HF_GOLD_SPLIT!r}). It is gated: set HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN, accept the dataset terms, or use --merge-ref-csv with a local copy.\n"
            f"Original error: {e}"
        ) from e
    gdf = gt.to_pandas() if hasattr(gt, "to_pandas") else pd.DataFrame(gt)
    for col in ("ID", "Reference_phn", "Annotation_phn"):
        if col not in gdf.columns:
            raise SystemExit(f"Gold dataset missing column {col!r}. Found: {list(gdf.columns)}")
    gdf = gdf[["ID", "Reference_phn", "Annotation_phn"]].drop_duplicates("ID", keep="first")
    out = df.merge(gdf, on="ID", how="left")
    out["phoneme_ref"] = out["Reference_phn"].map(lambda s: str(s).strip() if pd.notna(s) else "")
    out["phonemes"] = out["Annotation_phn"].map(lambda s: str(s).strip() if pd.notna(s) else "")
    out = out.drop(columns=["Reference_phn", "Annotation_phn"], errors="ignore")
    out["phoneme_aug"] = out["phonemes"]
    out["target_phoneme_sequence"] = out["phoneme_ref"].map(_norm_phone)
    out["num_target_phones"] = out["target_phoneme_sequence"].map(
        lambda s: 0 if not s else len(str(s).split())
    )
    return out


def _recompute_is_correct(out_df: pd.DataFrame) -> None:
    pr = out_df["phoneme_ref"].fillna("").map(_norm_phone)
    pn = out_df["phonemes"].fillna("").map(_norm_phone)
    out_df["is_correct"] = (pr == pn) & (pr != "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export IqraEval/QuranMB.v2 + IqraEval/IqraEval_Test_GT to wav/ + manifest."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory for manifest, metadata, wav/."
    )
    parser.add_argument(
        "--merge-ref-csv",
        default="",
        help="Optional CSV with ID + reference phonemes to merge (columns like phoneme_ref or Reference_phn).",
    )
    parser.add_argument(
        "--ref-phone-col",
        default="phoneme_ref",
        help="Preferred phoneme column name in --merge-ref-csv (fallbacks are tried if missing).",
    )
    parser.add_argument("--max-rows", type=int, default=-1, help="Only process first N rows (-1 = all).")
    parser.add_argument(
        "--skip-existing-wav",
        action="store_true",
        help="If a wav file already exists, do not rewrite it (still add row with duration from disk).",
    )
    args = parser.parse_args()

    _apply_hf_token_from_env()

    try:
        from datasets import load_dataset, Audio
    except Exception as e:  # pragma: no cover
        raise SystemExit("Install `datasets` (Hugging Face): pip install datasets") from e

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    meta_path = out_dir / "metadata.json"

    ds = load_dataset(HF_AUDIO, split=HF_AUDIO_SPLIT, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    n = len(ds)
    if args.max_rows > 0:
        n = min(n, int(args.max_rows))
        ds = ds.select(range(n))

    rows: list[dict] = []
    n_written = 0
    n_skipped_bytes = 0

    for i in tqdm(range(len(ds)), desc=f"{HF_AUDIO} -> wav + rows"):
        ex = ds[i]
        utt_id = str(ex["ID"])
        safe_name = utt_id.replace("/", "_")
        abs_wav = wav_dir / f"{safe_name}.wav"

        aud = ex.get("audio") or {}
        b = aud.get("bytes")
        if not b:
            n_skipped_bytes += 1
            continue

        duration_s = _wav_duration_sec_from_bytes(b)

        if not (args.skip_existing_wav and abs_wav.exists()):
            abs_wav.parent.mkdir(parents=True, exist_ok=True)
            abs_wav.write_bytes(b)
            n_written += 1
        elif abs_wav.exists():
            with wave.open(str(abs_wav), "rb") as w:
                duration_s = float(w.getnframes()) / float(w.getframerate() or 1)

        rel_str = _rel_to_repo(abs_wav)

        rows.append(
            {
                "ID": utt_id,
                "wav_path": rel_str,
                "wav_path_abs": str(abs_wav.resolve()),
                "text": "",
                "source": "quranmb_v2",
                "orig_split": HF_AUDIO_SPLIT,
                "is_correct": True,
                "raw_label": "quranmb_v2_huggingface",
                "speaker": "",
                "sentence_ref": "",
                "sentence_spoken": "",
                "tashkeel_sentence": "",
                "duration_s": float(duration_s),
                "phonemes": "",
                "phoneme_ref": "",
                "phoneme_aug": "",
            }
        )

    if not rows:
        raise SystemExit("No rows produced. Check dataset access and audio bytes in Parquet.")

    out_df = pd.DataFrame(rows)
    out_df = _apply_hf_gold(out_df)

    if args.merge_ref_csv:
        out_df = _merge_ref_phonemes(out_df, Path(args.merge_ref_csv), args.ref_phone_col)

    only_ref = (out_df["phonemes"].fillna("").str.len() == 0) & (out_df["phoneme_ref"].fillna("").str.len() > 0)
    if only_ref.any():
        out_df.loc[only_ref, "phonemes"] = out_df.loc[only_ref, "phoneme_ref"]
        out_df.loc[only_ref, "phoneme_aug"] = out_df.loc[only_ref, "phoneme_ref"]
    _recompute_is_correct(out_df)

    out_df = out_df.sort_values(["source", "orig_split", "ID"]).reset_index(drop=True)
    # Match column order of 01_build_mdd_bank_manifest train/dev output (keep_cols_mixed).
    col_order = [
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
        "is_correct",
        "phonemes",
        "phoneme_ref",
        "phoneme_aug",
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    out_df.to_csv(manifest_path, index=False)

    stats: dict = {
        "hf_audio_dataset": HF_AUDIO,
        "hf_audio_split": HF_AUDIO_SPLIT,
        "hf_gold_dataset": HF_GOLD,
        "hf_gold_split": HF_GOLD_SPLIT,
        "hf_token_set": bool((os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")).strip()),
        "manifest_csv": str(manifest_path),
        "metadata_json": str(meta_path),
        "output_dir": str(out_dir),
        "num_rows": int(len(out_df)),
        "num_wav_written_or_updated": int(n_written),
        "num_rows_missing_audio_bytes": int(n_skipped_bytes),
        "merge_ref_csv": str(args.merge_ref_csv) if args.merge_ref_csv else None,
        "rows_with_gold_ref_nontrivial": int((out_df["num_target_phones"] > 0).sum()) if "num_target_phones" in out_df.columns else 0,
    }
    stats.update(_phone_stats(out_df))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest: {manifest_path} ({len(out_df)} rows)")
    print(f"Saved metadata: {meta_path}")
    print(f"Wrote wav under:  {wav_dir} ({n_written} files written this run)")
    if (out_df["num_target_phones"] == 0).all():
        print(
            "Note: target_phoneme_sequence is empty for all rows. Set HF_TOKEN, accept "
            f"{HF_GOLD!r} on the Hub, or use --merge-ref-csv (ID, Reference_phn, …)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
