#!/usr/bin/env python3
"""Build phoneme-segment embeddings from aligned segment CSV.

Input presets by --dataset-set:
  - bank:  data/unified/bank/segments/bank_segments.csv
  - train: data/unified/train/segments/bank_segments.csv
  - dev:   data/unified/dev/segments/bank_segments.csv

The encoder outputs *frame-level* (downsampled) hidden states. One forward on the full
waveform per (batch of) utterances; for each target phoneme we **mean**-pool the hidden
frames that align to [start_sample, end_sample) and L2-normalize (one **phoneme-level**
vector per CSV row).

Output:
  - data/unified/mdd_segments/bank_embeddings.npy
  - data/unified/mdd_segments/bank_embeddings_meta.csv
  - data/unified/mdd_segments/bank_embeddings_stats.json
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTS_BY_SET = {
    "bank": REPO_ROOT / "data" / "unified" / "bank" / "segments" / "bank_segments.csv",
    "train": REPO_ROOT / "data" / "unified" / "train" / "segments" / "bank_segments.csv",
    "dev": REPO_ROOT / "data" / "unified" / "dev" / "segments" / "bank_segments.csv",
}
DEFAULT_OUTDIR_BY_SET = {
    "bank": REPO_ROOT / "data" / "unified" / "bank" / "embeddings",
    "train": REPO_ROOT / "data" / "unified" / "train" / "embeddings",
    "dev": REPO_ROOT / "data" / "unified" / "dev" / "embeddings",
}
HARD_NUM_PARTS = 3


def _as_abs(path_str: str) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_embed_model(model_id: str, device: str):
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for script 03.") from e
    feat = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return feat, model


def _prepare_waveform(seg: torch.Tensor) -> np.ndarray:
    """Mono float waveform in numpy."""
    if seg.dim() == 2:
        seg = seg.mean(dim=0)
    return seg.detach().cpu().numpy().astype(np.float32)


def _rec_in_bounds_wav(r: dict[str, Any], n_samples: int) -> bool:
    s, e = int(r["start_sample"]), int(r["end_sample"])
    e2 = min(e, n_samples)
    return 0 <= s and e2 > s and s < n_samples


def _prefix_sample_count_to_n_frames(
    model: Any, n_samples: torch.Tensor, device: str
) -> torch.Tensor:
    """Map prefix length in samples to cumulative frame count (same as HF conv stack)."""
    n = n_samples.to(device=device, dtype=torch.long)
    if n.dim() == 0:
        n = n.reshape(1)
    return model._get_feat_extract_output_lengths(n).to(torch.long).reshape(-1)


def _frame_indices_for_sample_span(
    model: Any, device: str, s: int, e: int, n_frames_utt: int
) -> tuple[int, int]:
    """[s, e) in samples → [fs, fe) frame index range for mean pooling (half-open, like HF conv)."""
    fs = int(_prefix_sample_count_to_n_frames(model, torch.tensor(s), device)[0].item())
    fe = int(_prefix_sample_count_to_n_frames(model, torch.tensor(e), device)[0].item())
    n = max(1, int(n_frames_utt))
    fe = min(fe, n)
    fs = min(max(0, fs), n)
    if fe <= fs:
        # Collapsed span: keep one valid frame
        fe = min(fs + 1, n)
    return fs, fe


def _pool_span_mean(
    hs_b: torch.Tensor, fs: int, fe: int
) -> torch.Tensor:  # hs_b: [T, D]
    if fe > fs:
        return hs_b[fs:fe, :].mean(dim=0)
    return hs_b[fs, :]


def _embed_utterance_batch(
    full_wav_np: list[np.ndarray],
    per_utt_recs: list[list[dict[str, Any]]],
    sampling_rate: int,
    feature_extractor: Any,
    model: Any,
    device: str,
) -> list[np.ndarray]:
    """One forward on padded batch of full utterances; return one embedding per (utt, rec) in order."""
    if not full_wav_np:
        return []
    inputs = feature_extractor(
        full_wav_np,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        out = model(input_values=input_values, attention_mask=attention_mask)
    hs = out.last_hidden_state  # [B, T, D]
    bsz, tmax, d = hs.shape
    in_lens = attention_mask.sum(dim=1).to(torch.long) if attention_mask is not None else torch.full(
        (bsz,), tmax, device=hs.device, dtype=torch.long
    )
    out_frame_counts = _prefix_sample_count_to_n_frames(model, in_lens, device)  # [B]

    embs: list[np.ndarray] = []
    for b in range(bsz):
        in_len = int(in_lens[b].item())
        n_f = int(out_frame_counts[b].item())
        n_f = min(max(1, n_f), tmax)
        hrow = hs[b, :n_f, :]  # valid prefix frames
        for rec in per_utt_recs[b]:
            s, e = int(rec["start_sample"]), int(rec["end_sample"])
            e = min(e, in_len)
            if e <= s or s < 0 or s >= in_len:
                raise ValueError("Invalid segment; caller must pre-filter segment bounds.")
            fs, fe = _frame_indices_for_sample_span(model, device, s, e, n_f)
            vec = _pool_span_mean(hrow, fs, fe)
            vec = torch.nn.functional.normalize(vec, p=2, dim=0)
            embs.append(vec.detach().cpu().numpy().astype(np.float32))
    return embs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build embedding bank from phoneme segments.")
    parser.add_argument(
        "--dataset-set",
        choices=["bank", "train", "dev", "custom"],
        default="bank",
        help="Preset IO location set. For custom, provide --segments-csv and --output-dir.",
    )
    parser.add_argument("--segments-csv", default="", help="Input segments CSV path (optional with preset set).")
    parser.add_argument("--output-dir", default="", help="Output directory path (optional with preset set).")
    parser.add_argument(
        "--embed-model-id",
        default="FatimahEmadEldin/wav2vec2-xls-r-300m-iqraeval",
        help="HF model for feature extraction (AutoModel).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Full utterances per model forward (batched, padded).",
    )
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--min-segment-ms", type=float, default=10.0)
    args = parser.parse_args()

    if args.dataset_set != "custom":
        default_in = DEFAULT_SEGMENTS_BY_SET[args.dataset_set]
        default_out = DEFAULT_OUTDIR_BY_SET[args.dataset_set]
        seg_csv = Path(args.segments_csv) if args.segments_csv else default_in
        out_dir = Path(args.output_dir) if args.output_dir else default_out
    else:
        if not args.segments_csv or not args.output_dir:
            raise SystemExit("--dataset-set custom requires --segments-csv and --output-dir")
        seg_csv = Path(args.segments_csv)
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not seg_csv.exists():
        raise SystemExit(f"Segments CSV not found: {seg_csv}")

    df = pd.read_csv(seg_csv)
    required = ["utt_id", "phoneme_index", "expected_phoneme", "start_sample", "end_sample", "utt_wav_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in segments CSV: {missing}")
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    # Filter extremely tiny segments
    if "duration_s" in df.columns:
        df = df[df["duration_s"] * 1000.0 >= args.min_segment_ms].copy()
    if df.empty:
        raise SystemExit("No rows left after segment filtering.")

    feature_extractor, model = _load_embed_model(args.embed_model_id, args.device)
    df = df.sort_values(["utt_id", "phoneme_index"], kind="merge")
    u_groups: list[pd.DataFrame] = [gg for _, gg in df.groupby("utt_id", sort=False)]
    if not u_groups:
        raise SystemExit("No utterances found after filtering.")
    utt_bsz = int(args.batch_size)

    if args.dataset_set == "dev" or args.dataset_set == "custom":
        HARD_NUM_PARTS = 1
    else:
        HARD_NUM_PARTS = 3

    for part_idx in range(HARD_NUM_PARTS):
        part_groups = u_groups[part_idx::HARD_NUM_PARTS]
        if not part_groups:
            continue
        part_suffix = f".part{part_idx + 1}of{HARD_NUM_PARTS}"
        out_emb = out_dir / f"bank_embeddings{part_suffix}.npy"
        out_meta = out_dir / f"bank_embeddings_meta{part_suffix}.csv"
        out_stats = out_dir / f"bank_embeddings_stats{part_suffix}.json"

        embeddings_chunks: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []
        n_fail = 0
        wav_cache: dict[str, tuple[torch.Tensor, int]] = {}
        n_rows_selected = int(sum(len(g) for g in part_groups))

        pbar = tqdm(
            range(0, len(part_groups), utt_bsz),
            desc=f"Embedding part {part_idx + 1}/{HARD_NUM_PARTS}",
        )
        for start in pbar:
            block = part_groups[start : start + utt_bsz]
            full_wav_np: list[np.ndarray] = []
            per_utt_recs: list[list[dict[str, Any]]] = []
            flat_meta: list[dict[str, Any]] = []
            for g in block:
                r0 = g.iloc[0]
                wav_path = str(r0["utt_wav_path"])
                wav_abs = _as_abs(wav_path)
                if not wav_abs.exists():
                    n_fail += len(g)
                    continue
                if wav_path not in wav_cache:
                    try:
                        w, sr = torchaudio.load(str(wav_abs))
                        if w.dim() == 2:
                            w = w.mean(dim=0, keepdim=True)
                        if sr != 16000:
                            w = torchaudio.functional.resample(w, sr, 16000)
                        wav_cache[wav_path] = (w, 16000)
                    except Exception:
                        n_fail += len(g)
                        continue
                wav, _ = wav_cache[wav_path]
                n_samp = int(wav.shape[1])
                vrecs: list[dict[str, Any]] = []
                for r in g.to_dict("records"):
                    if not _rec_in_bounds_wav(r, n_samp):
                        n_fail += 1
                    else:
                        vrecs.append(r)
                if not vrecs:
                    continue
                full_wav_np.append(_prepare_waveform(wav))
                per_utt_recs.append(vrecs)
                flat_meta.extend(vrecs)
            if not full_wav_np:
                continue
            embs_list = _embed_utterance_batch(
                full_wav_np,
                per_utt_recs,
                16000,
                feature_extractor,
                model,
                args.device,
            )
            embeddings_chunks.append(np.stack(embs_list, axis=0))
            meta_rows.extend(flat_meta)

        if not embeddings_chunks:
            print(f"[WARN] No embeddings produced for part {part_idx + 1}/{HARD_NUM_PARTS}")
            continue

        emb_all = np.concatenate(embeddings_chunks, axis=0)
        np.save(out_emb, emb_all)

        meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
        meta_df.insert(0, "embedding_index", np.arange(len(meta_df)))
        meta_df.to_csv(out_meta, index=False)

        phone_counter = Counter(meta_df["expected_phoneme"].tolist()) if "expected_phoneme" in meta_df.columns else Counter()
        stats = {
            "dataset_set": args.dataset_set,
            "num_parts": int(HARD_NUM_PARTS),
            "part_index": int(part_idx),
            "segments_csv": str(seg_csv),
            "embedding_model_id": args.embed_model_id,
            "device": args.device,
            "frame_pooling": "mean_of_hidden_frames_span",
            "utterance_batch_size": int(args.batch_size),
            "num_segments_input": int(n_rows_selected),
            "num_embeddings_output": int(emb_all.shape[0]),
            "embedding_dim": int(emb_all.shape[1]),
            "num_failed_segments": int(n_fail),
            "min_segment_ms": float(args.min_segment_ms),
            "unique_phonemes": int(len(phone_counter)),
            "top_phonemes": phone_counter.most_common(20),
            "embeddings_npy": str(out_emb),
            "metadata_csv": str(out_meta),
        }
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Saved embeddings: {out_emb}  shape={emb_all.shape}")
        print(f"Saved metadata:   {out_meta}")
        print(f"Saved stats:      {out_stats}")
        # Aggressively free part memory before next part.
        del embeddings_chunks, meta_rows, wav_cache, emb_all, meta_df
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

