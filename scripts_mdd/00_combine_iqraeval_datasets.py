#!/usr/bin/env python3
"""Combine IqraEval Hub datasets and save like data/raw/<split>: wav/ + metadata.csv.

Outputs (always, relative to the repo root):
  data/unified/wav/<ID>.wav
  data/unified/metadata.csv
  data/unified/stats.json

`metadata.csv` matches the Iqra_train-style export (see data/raw/train/metadata.csv):
  ID, wav_path, duration_s, text, phonemes, phoneme_ref, phoneme_aug
  plus: source, orig_split, is_correct, raw_label, speaker,
        sentence_ref, sentence_spoken, tashkeel_sentence

For mmap/NFS issues when loading a Hub cache, set --keep-in-memory or
HF_IQRA_FORCE_LOAD_IN_MEMORY=1 (needs enough RAM for that split).

Examples:
  python scripts/combine_iqraeval_datasets.py --print-schema-only
  python scripts/combine_iqraeval_datasets.py --max-per-source 100
  python scripts/combine_iqraeval_datasets.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = REPO_ROOT / "data" / "unified"
WAV_SUBDIR = "wav"

HF_IQRA_TRAIN = "IqraEval/Iqra_train"
HF_IQRA_TTS = "IqraEval/Iqra_TTS"
HF_EXTRA_IS26 = "IqraEval/Iqra_Extra_IS26"


def _norm_phoneme_str(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(tok for tok in str(s).strip().split() if tok and tok != "<sil>")


def _get_ci(row: Mapping[str, Any], *candidates: str) -> Any:
    lower = {k.lower(): k for k in row}
    for name in candidates:
        if name in row:
            return row[name]
        lk = name.lower()
        if lk in lower:
            return row[lower[lk]]
    return None


def _is_mis_label(label: str | None) -> bool:
    if label is None:
        return False
    s = str(label).strip().lower()
    if not s:
        return False
    if s in ("augmented", "aug", "mis", "misp", "mispronounced", "error", "wrong"):
        return True
    if s in (
        "reference",
        "ref",
        "correct",
        "original",
        "canonical",
        "clean",
        "ok",
        "match",
    ):
        return False
    if "aug" in s or "misp" in s or "error" in s or "wrong" in s:
        return True
    return False


def _safe_filename(id_str: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(id_str).strip(), flags=re.UNICODE)
    return s or "unnamed"


def _audio_to_wav(
    audio_data: Any, wav_path: Path, target_sr: int = 16000
) -> float | None:
    """Write normalized mono 16kHz wav; return duration in seconds, or None on failure.

    Prefer the same path as `src/data/download.py`: Hub `Audio(decode=False)` gives
    ``bytes`` (or a local ``path``), not a decoded object that may not be a ``dict``
    in recent ``datasets`` versions.
    """
    if audio_data is None:
        return None
    if isinstance(audio_data, Mapping) and not isinstance(audio_data, dict):
        try:
            audio_data = {k: audio_data[k] for k in list(audio_data.keys())}
        except Exception:
            return None
    if not isinstance(audio_data, dict):
        return None

    try:
        if audio_data.get("bytes"):
            rawb = audio_data["bytes"]
            suffix = Path(str(audio_data.get("path", "audio.mp3"))).suffix or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(rawb)
                tmp_path = tmp.name
            try:
                t, sr = torchaudio.load(tmp_path)
            finally:
                os.remove(tmp_path)
            t = t.mean(0, keepdim=True)
        elif audio_data.get("path"):
            t, sr = torchaudio.load(str(audio_data["path"]))
            t = t.mean(0, keepdim=True)
        elif "array" in audio_data and audio_data["array"] is not None:
            arr = np.asarray(audio_data["array"], dtype=np.float32)
            sr = int(audio_data.get("sampling_rate") or 16000)
            t = torch.from_numpy(arr)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            else:
                t = t.mean(dim=0, keepdim=True)
        else:
            return None
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)
            sr = target_sr
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(wav_path), t, sr)
        return float(t.shape[1] / sr)
    except Exception as e:  # pragma: no cover
        print(f"  [WARN] write wav {wav_path}: {e}", file=sys.stderr)
        return None


def _map_iqra_train_row(row: Mapping[str, Any], orig_split: str) -> dict[str, Any]:
    uid = str(_get_ci(row, "id") or "").strip() or "unknown"
    rid = f"iqra_train_{orig_split}_{uid}"
    pr = _norm_phoneme_str(_get_ci(row, "phoneme_ref", "Phoneme_ref"))
    paug = _norm_phoneme_str(_get_ci(row, "phoneme_aug"))
    if not paug:
        paug = pr
    return {
        "id": rid,
        "source": "iqra_train",
        "orig_split": orig_split,
        "reference_phoneme": pr,
        "spoken_phoneme": pr,
        "phoneme_aug": paug,
        "sentence": str(_get_ci(row, "sentence") or ""),
        "sentence_ref": str(_get_ci(row, "sentence") or ""),
        "sentence_spoken": str(_get_ci(row, "sentence") or ""),
        "tashkeel_sentence": str(_get_ci(row, "tashkeel_sentence", "tashkeel") or ""),
        "is_correct": True,
        "raw_label": "correct_speech",
        "speaker": "",
        "audio": row["audio"],
    }


def _map_iqra_tts_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    label_raw = _get_ci(row, "label", "Label")
    raw_l = "" if label_raw is None else str(label_raw)
    is_mis = _is_mis_label(raw_l if raw_l else None)
    sref = str(_get_ci(row, "sentence_ref", "sentence", "ref") or "")
    saug = str(_get_ci(row, "sentence_aug", "sentence_augmented") or "")
    pr = _norm_phoneme_str(_get_ci(row, "phoneme_ref", "phonemeRef"))
    pmis = _norm_phoneme_str(_get_ci(row, "phoneme_mis", "phoneme_aug", "Annotation_phoneme"))
    if is_mis and pmis:
        spoken = pmis
        sentence_spoken = saug if saug else sref
    else:
        spoken = pr
        sentence_spoken = sref
    spk = _get_ci(row, "speaker", "Speaker")
    speaker = "" if spk is None else str(spk)
    rid = f"iqra_tts_train_{index:07d}"
    paug = pmis if (is_mis and pmis) else pr
    return {
        "id": rid,
        "source": "iqra_tts",
        "orig_split": "train",
        "reference_phoneme": pr,
        "spoken_phoneme": spoken,
        "phoneme_aug": paug,
        "sentence": sref if sref else saug,
        "sentence_ref": sref,
        "sentence_spoken": sentence_spoken,
        "tashkeel_sentence": "",
        "is_correct": not is_mis,
        "raw_label": raw_l,
        "speaker": speaker,
        "audio": row["audio"],
    }


def _map_extra_is26_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    label_raw = _get_ci(row, "label", "Label", "split", "kind", "type")
    raw_l = "" if label_raw is None else str(label_raw)
    sref = str(
        _get_ci(row, "sentence_ref", "sentence", "text", "tashkeel_sentence") or ""
    )
    saug = str(_get_ci(row, "sentence_aug", "sentence_spoken", "verbal") or "")
    pr = _norm_phoneme_str(
        _get_ci(row, "phoneme_ref", "reference_phoneme", "Reference_phn", "phones")
    )
    pmis = _norm_phoneme_str(
        _get_ci(row, "phoneme_mis", "phoneme_aug", "Annotation_phn", "spoken_phoneme")
    )
    if raw_l and _is_mis_label(raw_l):
        is_mis = True
    elif raw_l and not _is_mis_label(raw_l):
        is_mis = False
    else:
        is_mis = pr != pmis
    if is_mis and pmis:
        spoken = pmis
        sentence_spoken = saug if saug else sref
    else:
        spoken = pr
        sentence_spoken = sref
    if not raw_l:
        raw_l = "inferred_mis" if is_mis else "inferred_ref"
    spk = _get_ci(row, "speaker", "Speaker", "reciter")
    speaker = "" if spk is None else str(spk)
    uid = _get_ci(row, "id", "ID", "utt_id", "name")
    uid_s = f"{index:07d}" if uid is None else str(uid).replace(" ", "_")
    rid = f"iqra_extra_is26_train_{uid_s}"
    tashk = str(_get_ci(row, "tashkeel_sentence", "tashkeel") or "")
    paug = pmis if is_mis and pmis else pr
    return {
        "id": rid,
        "source": "iqra_extra_is26",
        "orig_split": "train",
        "reference_phoneme": pr,
        "spoken_phoneme": spoken,
        "phoneme_aug": paug,
        "sentence": sref if sref else saug,
        "sentence_ref": sref,
        "sentence_spoken": sentence_spoken,
        "tashkeel_sentence": tashk,
        "is_correct": not is_mis,
        "raw_label": raw_l,
        "speaker": speaker,
        "audio": row["audio"],
    }


def _unified_to_metadata_row(
    u: dict[str, Any], wav_path_csv: str, duration_s: float
) -> dict[str, Any]:
    """One CSV row: same style as data/raw/train (relative wav_path under repo)."""
    ref = u["reference_phoneme"]
    spk = u["spoken_phoneme"]
    aug = u.get("phoneme_aug", spk)
    if not aug:
        aug = ref
    return {
        "ID": u["id"],
        "wav_path": wav_path_csv,
        "duration_s": round(duration_s, 3),
        "text": u["sentence"],
        "phonemes": spk,
        "phoneme_ref": ref,
        "phoneme_aug": aug,
        "source": u["source"],
        "orig_split": u["orig_split"],
        "is_correct": u["is_correct"],
        "raw_label": u["raw_label"],
        "speaker": u.get("speaker", "") or "",
        "sentence_ref": u.get("sentence_ref", ""),
        "sentence_spoken": u.get("sentence_spoken", ""),
        "tashkeel_sentence": u.get("tashkeel_sentence", "") or "",
    }


def _print_one_example(name: str, path: str, split: str, token: str | None) -> None:
    print(f"\n=== {name} ({path} split={split}) ===", flush=True)
    try:
        ex = next(
            iter(
                load_dataset(
                    path,
                    split=split,
                    **({"token": token} if token else {}),
                    streaming=True,
                )
            )
        )
    except Exception as e:
        print(f"  ERROR: {e!r}", flush=True)
        return
    keys = list(ex.keys()) if ex else []
    print("keys:", keys, flush=True)
    for k in keys:
        v = ex[k]
        tname = type(v).__name__
        if isinstance(v, dict) and (
            "array" in v or "path" in v or "bytes" in v
        ):
            print(f"  {k}: {tname} (audio/blob)", flush=True)
        else:
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {k}: {tname} = {s!r}", flush=True)


def _load_kwargs(token: str | None, keep_in_memory: bool) -> dict[str, Any]:
    """Parquet-based Hub sets do not need trust_remote_code (deprecated warning in 2025+)."""
    kw: dict[str, Any] = {"keep_in_memory": keep_in_memory}
    if token:
        kw["token"] = token
    return kw


def _slice_ds(ds, max_samples: int):
    n = len(ds) if max_samples < 0 else min(max_samples, len(ds))
    return ds if n == len(ds) else ds.select(range(n))


def _cast_audio_no_decode_for_export(ds):
    """Keep bytes on disk (or path); avoids decoded types that are not ``dict``."""
    if "audio" in ds.column_names:
        return ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    return ds


def export_unified(
    out_dir: Path,
    max_samples: int,
    token: str | None,
    no_dev: bool,
    skip_train: bool,
    skip_tts: bool,
    skip_extra: bool,
    keep_in_memory: bool,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    wav_dir = out_dir / WAV_SUBDIR
    wav_dir.mkdir(parents=True, exist_ok=True)

    lkw = _load_kwargs(token, keep_in_memory)
    env_ki = os.environ.get("HF_IQRA_FORCE_LOAD_IN_MEMORY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if env_ki:
        lkw["keep_in_memory"] = True

    rows: list[dict[str, Any]] = []
    total_dur = 0.0
    phoneme_counter: Counter[str] = Counter()
    n_ok = 0
    n_skip = 0
    def _path_for_csv(wav_file: Path) -> str:
        return f"data/unified/{WAV_SUBDIR}/{wav_file.name}"

    if not skip_train:
        for split in (["train"] + ([] if no_dev else ["dev"])):
            raw = load_dataset(HF_IQRA_TRAIN, split=split, **lkw)
            raw = _cast_audio_no_decode_for_export(raw)
            raw = _slice_ds(raw, max_samples)
            for i in tqdm(
                range(len(raw)),
                desc=f"Iqra_train {split} → disk",
            ):
                u = _map_iqra_train_row(raw[i], split)
                fn = f"{_safe_filename(u['id'])}.wav"
                wpath = wav_dir / fn
                d = _audio_to_wav(u["audio"], wpath)
                if d is None:
                    n_skip += 1
                    continue
                total_dur += d
                n_ok += 1
                m = _unified_to_metadata_row(u, _path_for_csv(wpath), d)
                for p in m["phonemes"].split():
                    if p and p != "<sil>":
                        phoneme_counter[p] += 1
                rows.append(m)

    if not skip_tts:
        raw = load_dataset(HF_IQRA_TTS, split="train", **lkw)
        raw = _cast_audio_no_decode_for_export(raw)
        raw = _slice_ds(raw, max_samples)
        for i in tqdm(range(len(raw)), desc="Iqra_TTS → disk"):
            u = _map_iqra_tts_row(raw[i], i)
            fn = f"{_safe_filename(u['id'])}.wav"
            wpath = wav_dir / fn
            d = _audio_to_wav(u["audio"], wpath)
            if d is None:
                n_skip += 1
                continue
            total_dur += d
            n_ok += 1
            m = _unified_to_metadata_row(u, _path_for_csv(wpath), d)
            for p in m["phonemes"].split():
                if p and p != "<sil>":
                    phoneme_counter[p] += 1
            rows.append(m)

    if not skip_extra:
        raw = load_dataset(HF_EXTRA_IS26, split="train", **lkw)
        raw = _cast_audio_no_decode_for_export(raw)
        raw = _slice_ds(raw, max_samples)
        for i in tqdm(range(len(raw)), desc="Iqra_Extra_IS26 → disk"):
            u = _map_extra_is26_row(raw[i], i)
            fn = f"{_safe_filename(u['id'])}.wav"
            wpath = wav_dir / fn
            d = _audio_to_wav(u["audio"], wpath)
            if d is None:
                n_skip += 1
                continue
            total_dur += d
            n_ok += 1
            m = _unified_to_metadata_row(u, _path_for_csv(wpath), d)
            for p in m["phonemes"].split():
                if p and p != "<sil>":
                    phoneme_counter[p] += 1
            rows.append(m)

    if not rows:
        raise SystemExit("No rows exported (all skipped or all sources disabled).")

    # Mirror train: primary columns first, then extra
    col_order = [
        "ID",
        "wav_path",
        "duration_s",
        "text",
        "phonemes",
        "phoneme_ref",
        "phoneme_aug",
        "source",
        "orig_split",
        "is_correct",
        "raw_label",
        "speaker",
        "sentence_ref",
        "sentence_spoken",
        "tashkeel_sentence",
    ]
    df = pd.DataFrame(rows)[col_order]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    by_source: dict[str, int] = dict(Counter(df["source"]))

    stats: dict[str, Any] = {
        "export": "unified",
        "output_dir": str(out_dir),
        "num_rows": len(df),
        "num_skipped": n_skip,
        "total_duration_hours": round(total_dur / 3600, 2),
        "unique_phonemes": len(phoneme_counter),
        "top_phonemes": phoneme_counter.most_common(12),
        "by_source": by_source,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine Iqra_train, Iqra_TTS, Iqra_Extra_IS26; save to data/unified/ (wav + metadata.csv)."
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=-1,
        help="Max rows per source after loading (-1 = all).",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Do not include Iqra_train 'dev' split.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Omit IqraEval/Iqra_train.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Omit IqraEval/Iqra_TTS.",
    )
    parser.add_argument(
        "--skip-extra",
        action="store_true",
        help="Omit IqraEval/Iqra_Extra_IS26.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Hugging Face token (or HF_TOKEN / HUGGING_FACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--print-schema-only",
        action="store_true",
        help="Print one example's keys from each source and exit.",
    )
    parser.add_argument(
        "--keep-in-memory",
        action="store_true",
        help="load_dataset(keep_in_memory=True); helps some NFS/mmap issues (needs RAM).",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.print_schema_only:
        for name, path, sp in [
            ("Iqra_train", HF_IQRA_TRAIN, "train"),
            ("Iqra_train dev", HF_IQRA_TRAIN, "dev"),
            ("Iqra_TTS", HF_IQRA_TTS, "train"),
            ("Iqra_Extra_IS26", HF_EXTRA_IS26, "train"),
        ]:
            _print_one_example(name, path, sp, token)
        return 0

    stats = export_unified(
        out_dir=UNIFIED_DIR,
        max_samples=args.max_per_source,
        token=token,
        no_dev=args.no_dev,
        skip_train=args.skip_train,
        skip_tts=args.skip_tts,
        skip_extra=args.skip_extra,
        keep_in_memory=args.keep_in_memory,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(
        f"Wrote {UNIFIED_DIR / 'metadata.csv'} ({stats['num_rows']} rows) and wav under {UNIFIED_DIR / 'wav'}/",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
