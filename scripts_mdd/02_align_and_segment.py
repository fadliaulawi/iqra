#!/usr/bin/env python3
"""Align utterances to target phonemes and export phoneme-level segments.

Supported backends:
  - uniform: split utterance duration uniformly across target phonemes.
  - ctc_forced: CTC forced alignment using a Transformers-compatible CTC model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_BY_SET = {
    "bank": REPO_ROOT / "data" / "unified" / "bank" / "manifest.csv",
    "train": REPO_ROOT / "data" / "unified" / "train" / "manifest.csv",
    "dev": REPO_ROOT / "data" / "unified" / "dev" / "manifest.csv",
}
DEFAULT_OUTDIR_BY_SET = {
    "bank": REPO_ROOT / "data" / "unified" / "bank" / "segments",
    "train": REPO_ROOT / "data" / "unified" / "train" / "segments",
    "dev": REPO_ROOT / "data" / "unified" / "dev" / "segments",
}


def _as_abs_wav(path_str: str) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else (REPO_ROOT / p)


def _normalize_audio(wav: torch.Tensor, sr: int, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def _uniform_boundaries(total_samples: int, n_parts: int) -> list[tuple[int, int]]:
    if n_parts <= 0:
        return []
    if total_samples <= 0:
        return [(0, 0)] * n_parts
    edges = [int(round(i * total_samples / n_parts)) for i in range(n_parts + 1)]
    spans: list[tuple[int, int]] = []
    for i in range(n_parts):
        s, e = edges[i], edges[i + 1]
        if e < s:
            e = s
        spans.append((s, e))
    return spans


def _load_ctc_aligner(model_id: str, device: str, processor_id: str | None = None):
    try:
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoProcessor, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for --backend ctc_forced. Install it first."
        ) from e

    proc_src = processor_id or model_id
    try:
        processor = AutoProcessor.from_pretrained(proc_src)
    except Exception:
        try:
            feat = AutoFeatureExtractor.from_pretrained(proc_src)
            tok = AutoTokenizer.from_pretrained(proc_src)

            class _Proc:
                def __init__(self, f, t):
                    self.feature_extractor = f
                    self.tokenizer = t

                def __call__(self, audio, sampling_rate, return_tensors="pt"):
                    return self.feature_extractor(
                        audio, sampling_rate=sampling_rate, return_tensors=return_tensors
                    )

            processor = _Proc(feat, tok)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load processor/tokenizer from '{proc_src}'."
            ) from e

    try:
        model = AutoModelForCTC.from_pretrained(model_id).to(device)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load CTC model from '{model_id}'. Need a Transformers-compatible CTC repo."
        ) from e
    model.eval()

    vocab = processor.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    blank_id = (
        getattr(processor.tokenizer, "pad_token_id", None)
        if getattr(processor.tokenizer, "pad_token_id", None) is not None
        else 0
    )
    return processor, model, vocab, inv_vocab, blank_id


def _map_phones_to_ids(phones: list[str], vocab: dict[str, int]) -> list[int] | None:
    ids: list[int] = []
    for ph in phones:
        if ph in vocab:
            ids.append(vocab[ph])
            continue
        ph2 = ph.strip()
        if ph2 in vocab:
            ids.append(vocab[ph2])
            continue
        return None
    return ids


def _ctc_viterbi_state_path(log_probs: torch.Tensor, token_ids: list[int], blank_id: int) -> list[int] | None:
    t_steps = int(log_probs.shape[0])
    if t_steps <= 0 or not token_ids:
        return None

    states: list[int] = [blank_id]
    for tid in token_ids:
        states.append(tid)
        states.append(blank_id)
    s_count = len(states)

    neg_inf = -1e30
    dp = torch.full((t_steps, s_count), neg_inf, dtype=log_probs.dtype, device=log_probs.device)
    bp = torch.full((t_steps, s_count), -1, dtype=torch.long, device=log_probs.device)

    dp[0, 0] = log_probs[0, states[0]]
    if s_count > 1:
        dp[0, 1] = log_probs[0, states[1]]
        bp[0, 1] = 0

    # Vectorized DP transition (replaces Python loop over states).
    # Candidates:
    #   0) stay at s
    #   1) step from s-1
    #   2) skip from s-2 (only for non-blank states that differ from s-2)
    states_t = torch.tensor(states, device=log_probs.device, dtype=torch.long)
    s_idx = torch.arange(s_count, device=log_probs.device, dtype=torch.long)
    neg_inf_t = torch.tensor(neg_inf, device=log_probs.device, dtype=log_probs.dtype)

    valid_skip = torch.zeros(s_count, device=log_probs.device, dtype=torch.bool)
    if s_count > 2:
        # skip allowed only if current state is non-blank and different from s-2
        valid_skip[2:] = (states_t[2:] != blank_id) & (states_t[2:] != states_t[:-2])

    for t in range(1, t_steps):
        prev = dp[t - 1]

        stay = prev

        step = torch.full_like(prev, neg_inf_t)
        step[1:] = prev[:-1]

        skip = torch.full_like(prev, neg_inf_t)
        if s_count > 2:
            skip[2:] = prev[:-2]
            skip = torch.where(valid_skip, skip, neg_inf_t)

        cand = torch.stack([stay, step, skip], dim=0)  # [3, S]
        best_val, best_kind = torch.max(cand, dim=0)  # each [S]

        # map best kind -> previous state index
        prev_choices = torch.stack([s_idx, s_idx - 1, s_idx - 2], dim=0)  # [3, S]
        best_prev = prev_choices.gather(0, best_kind.unsqueeze(0)).squeeze(0)

        dp[t] = best_val + log_probs[t, states_t]
        bp[t] = best_prev

    end_cands = [(dp[t_steps - 1, s_count - 1], s_count - 1)]
    if s_count - 2 >= 0:
        end_cands.append((dp[t_steps - 1, s_count - 2], s_count - 2))
    _, s = max(end_cands, key=lambda x: float(x[0]))

    state_path = [0] * t_steps
    for t in range(t_steps - 1, -1, -1):
        state_path[t] = int(s)
        if t > 0:
            s = int(bp[t, s])
            if s < 0:
                return None
    return state_path


def _ctc_forced_boundaries(wav: torch.Tensor, sr: int, phones: list[str], aligner: dict[str, Any]) -> list[tuple[int, int]] | None:
    processor = aligner["processor"]
    model = aligner["model"]
    vocab = aligner["vocab"]
    blank_id = aligner["blank_id"]
    device = aligner["device"]

    token_ids = _map_phones_to_ids(phones, vocab)
    if token_ids is None:
        return None

    audio_np = wav.squeeze(0).cpu().numpy()
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.inference_mode():
        logits = model(input_values).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    state_path = _ctc_viterbi_state_path(log_probs, token_ids, blank_id)
    if state_path is None:
        return None

    t_steps = len(state_path)
    if t_steps == 0:
        return None
    samples_per_frame = wav.shape[1] / float(t_steps)
    spans: list[tuple[int, int]] = []
    for k in range(len(token_ids)):
        target_state = 2 * k + 1
        frames = [i for i, s in enumerate(state_path) if s == target_state]
        if not frames:
            return None
        f0, f1 = min(frames), max(frames)
        s0 = int(round(f0 * samples_per_frame))
        s1 = int(round((f1 + 1) * samples_per_frame))
        if s1 < s0:
            s1 = s0
        spans.append((s0, s1))
    return spans


def _ctc_forced_boundaries_batch(
    wavs: list[torch.Tensor],
    sr: int,
    phones_batch: list[list[str]],
    aligner: dict[str, Any],
) -> list[list[tuple[int, int]] | None]:
    """Batch version of CTC forced alignment.

    Returns one spans-list (or None) per utterance.
    """
    processor = aligner["processor"]
    model = aligner["model"]
    vocab = aligner["vocab"]
    blank_id = aligner["blank_id"]
    device = aligner["device"]

    if not wavs:
        return []

    token_ids_batch: list[list[int] | None] = [
        _map_phones_to_ids(phones, vocab) for phones in phones_batch
    ]

    audios = [w.squeeze(0).cpu().numpy() for w in wavs]
    try:
        inputs = processor(
            audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        # fallback if processor wrapper doesn't expose padding kwarg
        inputs = processor(audios, sampling_rate=sr, return_tensors="pt")

    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        out = model(input_values=input_values, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)

    # estimate valid output length per sample
    bsz, t_max, _ = log_probs.shape
    if attention_mask is not None and hasattr(model, "_get_feat_extract_output_lengths"):
        in_lens = attention_mask.sum(dim=1).to(torch.long)
        out_lens = model._get_feat_extract_output_lengths(in_lens).to(torch.long)
        out_lens = torch.clamp(out_lens, min=1, max=t_max)
    else:
        out_lens = torch.full((bsz,), t_max, dtype=torch.long, device=log_probs.device)

    all_spans: list[list[tuple[int, int]] | None] = []
    for i in range(bsz):
        token_ids = token_ids_batch[i]
        if token_ids is None:
            all_spans.append(None)
            continue
        t_len = int(out_lens[i].item())
        lp = log_probs[i, :t_len, :]
        state_path = _ctc_viterbi_state_path(lp, token_ids, blank_id)
        if state_path is None:
            all_spans.append(None)
            continue
        if len(state_path) == 0:
            all_spans.append(None)
            continue
        samples_per_frame = wavs[i].shape[1] / float(len(state_path))
        spans: list[tuple[int, int]] = []
        ok = True
        for k in range(len(token_ids)):
            target_state = 2 * k + 1
            frames = [t for t, s in enumerate(state_path) if s == target_state]
            if not frames:
                ok = False
                break
            f0, f1 = min(frames), max(frames)
            s0 = int(round(f0 * samples_per_frame))
            s1 = int(round((f1 + 1) * samples_per_frame))
            if s1 < s0:
                s1 = s0
            spans.append((s0, s1))
        all_spans.append(spans if ok else None)
    return all_spans


def main() -> int:
    parser = argparse.ArgumentParser(description="Create phoneme-level segments from bank/train/dev manifest.")
    parser.add_argument(
        "--dataset-set",
        choices=["bank", "train", "dev", "custom"],
        default="bank",
        help="Preset IO location set. For custom, provide --input-manifest and --output-dir.",
    )
    parser.add_argument("--input-manifest", default="", help="Input manifest CSV path. Optional when using preset set.")
    parser.add_argument("--output-dir", default="", help="Output dir for segment CSV/stats. Optional when using preset set.")
    parser.add_argument("--backend", default="uniform", choices=["uniform", "ctc_forced"], help="Alignment backend.")
    parser.add_argument("--ctc-model-id", default="IqraEval/Iqra_wav2vec2_base", help="HF model id for CTC forced alignment backend.")
    parser.add_argument("--ctc-processor-id", default="", help="Optional separate HF processor/tokenizer repo id.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Device used by CTC backend.")
    parser.add_argument("--save-segment-wavs", action="store_true", help="Write per-phoneme wav files under output-dir/wav_segments.")
    parser.add_argument("--max-rows", type=int, default=-1, help="Limit utterances for fast testing (-1=all).")
    parser.add_argument("--min-segment-ms", type=float, default=10.0, help="Discard segments shorter than this duration.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used for ctc_forced backend.")
    args = parser.parse_args()

    if args.dataset_set != "custom":
        default_in = DEFAULT_MANIFEST_BY_SET[args.dataset_set]
        default_out = DEFAULT_OUTDIR_BY_SET[args.dataset_set]
        in_csv = Path(args.input_manifest) if args.input_manifest else default_in
        out_dir = Path(args.output_dir) if args.output_dir else default_out
    else:
        if not args.input_manifest or not args.output_dir:
            raise SystemExit("--dataset-set custom requires --input-manifest and --output-dir")
        in_csv = Path(args.input_manifest)
        out_dir = Path(args.output_dir)

    out_csv = out_dir / "bank_segments.csv"
    out_stats = out_dir / "bank_segments_stats.json"
    seg_wav_dir = out_dir / "wav_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_segment_wavs:
        seg_wav_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise SystemExit(f"Input manifest not found: {in_csv}")

    df = pd.read_csv(in_csv)
    required = ["ID", "target_phoneme_sequence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input manifest missing columns: {missing}")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    rows: list[dict] = []
    n_total = len(df)
    n_failed = 0
    n_short = 0
    phone_counter: Counter[str] = Counter()
    n_ctc_fallback_uniform = 0

    aligner: dict[str, Any] | None = None
    if args.backend == "ctc_forced":
        aligner = {}
        processor, model, vocab, inv_vocab, blank_id = _load_ctc_aligner(
            args.ctc_model_id, args.device, processor_id=args.ctc_processor_id or None
        )
        aligner["processor"] = processor
        aligner["model"] = model
        aligner["vocab"] = vocab
        aligner["inv_vocab"] = inv_vocab
        aligner["blank_id"] = blank_id
        aligner["device"] = args.device

    records = df.to_dict("records")
    step = args.batch_size if args.backend == "ctc_forced" else 1
    for start in tqdm(range(0, len(records), step), total=(len(records) + step - 1) // step, desc=f"Align+segment [{args.backend}]"):
        batch_recs = records[start : start + step]
        valid_items: list[tuple[dict[str, Any], torch.Tensor, list[str], Path, int]] = []
        batch_wavs: list[torch.Tensor] = []
        batch_phones: list[list[str]] = []

        # load/prep audio for the batch
        for rec in batch_recs:
            utt_id = str(rec["ID"])
            wav_path = rec.get("wav_path_abs") or rec.get("wav_path")
            if not wav_path:
                n_failed += 1
                continue
            wav_abs = _as_abs_wav(wav_path)
            if not wav_abs.exists():
                n_failed += 1
                continue

            phones = [p for p in str(rec["target_phoneme_sequence"]).split() if p]
            if not phones:
                n_failed += 1
                continue

            try:
                wav, sr = torchaudio.load(str(wav_abs))
                wav, sr = _normalize_audio(wav, sr, target_sr=16000)
            except Exception:
                n_failed += 1
                continue

            valid_items.append((rec, wav, phones, wav_abs, sr))
            if args.backend == "ctc_forced":
                batch_wavs.append(wav)
                batch_phones.append(phones)

        if not valid_items:
            continue

        # compute spans (batched for ctc_forced, per-utt for uniform)
        spans_batch: list[list[tuple[int, int]] | None] = []
        if args.backend == "ctc_forced":
            # all normalized to 16k, so sample rate shared
            spans_batch = _ctc_forced_boundaries_batch(
                batch_wavs,
                16000,
                batch_phones,
                aligner,  # type: ignore[arg-type]
            )
        else:
            for _, wav, phones, _, _ in valid_items:
                spans_batch.append(_uniform_boundaries(int(wav.shape[1]), len(phones)))
        for item_idx, (rec, wav, phones, wav_abs, sr) in enumerate(valid_items):
            utt_id = str(rec["ID"])
            total_samples = int(wav.shape[1])
            spans = spans_batch[item_idx]
            if spans is None or len(spans) != len(phones):
                spans = _uniform_boundaries(total_samples, len(phones))
                if args.backend == "ctc_forced":
                    n_ctc_fallback_uniform += 1

            for idx, (ph, (s, e)) in enumerate(zip(phones, spans)):
                seg = wav[:, s:e]
                dur_s = float((e - s) / sr) if e > s else 0.0
                if dur_s * 1000.0 < args.min_segment_ms:
                    n_short += 1
                    continue

                seg_rel = ""
                if args.save_segment_wavs:
                    seg_name = f"{utt_id}__p{idx:03d}__{ph}.wav".replace("/", "_")
                    seg_path = seg_wav_dir / seg_name
                    try:
                        torchaudio.save(str(seg_path), seg, sr)
                        seg_rel = str(seg_path.relative_to(REPO_ROOT)) if seg_path.is_absolute() else str(seg_path)
                    except Exception:
                        seg_rel = ""

                rows.append({
                    "utt_id": utt_id,
                    "phoneme_index": idx,
                    "expected_phoneme": ph,
                    "start_sample": int(s),
                    "end_sample": int(e),
                    "start_sec": float(s / sr),
                    "end_sec": float(e / sr),
                    "duration_s": dur_s,
                    "utt_wav_path": str(wav_abs),
                    "segment_wav_path": seg_rel,
                    "source": rec.get("source", ""),
                    "orig_split": rec.get("orig_split", ""),
                    # Optional passthrough labels for downstream retrieval metrics.
                    "is_correct": rec.get("is_correct", ""),
                    "phonemes": rec.get("phonemes", ""),
                    "phoneme_ref": rec.get("phoneme_ref", ""),
                    "phoneme_aug": rec.get("phoneme_aug", ""),
                })
                phone_counter[ph] += 1

    if not rows:
        raise SystemExit("No segment rows produced. Check input manifest and wav paths.")

    seg_df = pd.DataFrame(rows).sort_values(["utt_id", "phoneme_index"]).reset_index(drop=True)
    seg_df.to_csv(out_csv, index=False)

    stats = {
        "dataset_set": args.dataset_set,
        "input_manifest": str(in_csv),
        "output_csv": str(out_csv),
        "backend": args.backend,
        "save_segment_wavs": bool(args.save_segment_wavs),
        "num_utterances_input": int(n_total),
        "num_segments_output": int(len(seg_df)),
        "num_failed_utterances": int(n_failed),
        "num_skipped_short_segments": int(n_short),
        "num_ctc_fallback_to_uniform": int(n_ctc_fallback_uniform),
        "unique_phonemes": int(len(phone_counter)),
        "top_phonemes": phone_counter.most_common(20),
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved segments CSV: {out_csv}")
    print(f"Saved stats JSON:   {out_stats}")
    if args.save_segment_wavs:
        print(f"Saved segment wavs under: {seg_wav_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
