"""Build unified Tajweed metadata, auto split train/dev, and optionally link audio.

Default behavior samples 3 verses per surah (114 surahs) with a fixed seed,
then phonemizes each ref with quranic-phonemizer.

Output is text/label metadata (no audio paths):
  ID,quran_ref,surah,ayah,wav_path,duration_s,text,phones_tajweed,phonemes,tajweed_match_score
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from quranic_phonemizer import Phonemizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "quranic-phonemizer is required. Install with: pip install -r requirements_tajweed.txt"
    ) from exc


QURAN_VERSE_COUNTS = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98,
    20: 135, 21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88,
    29: 69, 30: 60, 31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88,
    39: 75, 40: 85, 41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29,
    49: 18, 50: 45, 51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22,
    59: 24, 60: 13, 61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52,
    69: 52, 70: 44, 71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40,
    79: 46, 80: 42, 81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26,
    89: 30, 90: 20, 91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8,
    99: 8, 100: 11, 101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3,
    109: 6, 110: 3, 111: 5, 112: 4, 113: 5, 114: 6,
}


def _norm_phone_str(s: str) -> str:
    return " ".join(tok for tok in str(s).strip().split() if tok and tok != "<sil>")


def _build_audio_path(audio_root: Path, quran_ref: str, ext: str) -> Path:
    surah_s, ayah_s = str(quran_ref).split(":")
    return audio_root / f"{int(surah_s)}-{int(ayah_s)}.{ext}"


def attach_audio(df: pd.DataFrame, audio_root: str, ext: str = "m4a") -> tuple[pd.DataFrame, int]:
    """Attach wav_path using quran_ref -> '<surah>-<ayah>.<ext>'."""
    audio_dir = Path(audio_root)
    if not audio_dir.exists():
        raise ValueError(f"audio_root not found: {audio_root}")
    wav_paths = []
    durations = []
    missing = 0
    for ref in df["quran_ref"].astype(str):
        p = _build_audio_path(audio_dir, ref, ext=ext)
        if p.exists():
            wav_paths.append(str(p))
            durations.append(0.0)
        else:
            wav_paths.append("")
            durations.append(0.0)
            missing += 1
    out = df.copy()
    out["wav_path"] = wav_paths
    out["duration_s"] = durations
    return out, missing


def sample_refs(verses_per_surah: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    refs: list[str] = []
    for surah in range(1, 115):
        max_ayah = QURAN_VERSE_COUNTS[surah]
        n = min(verses_per_surah, max_ayah)
        ayahs = sorted(rng.sample(range(1, max_ayah + 1), n))
        refs.extend([f"{surah}:{ayah}" for ayah in ayahs])
    return refs


def build_unified_metadata(verses_per_surah: int, seed: int) -> pd.DataFrame:
    refs = sample_refs(verses_per_surah=verses_per_surah, seed=seed)
    pm = Phonemizer()
    rows = []
    for ref in tqdm(refs, desc="Phonemize refs"):
        res = pm.phonemize(ref=ref)
        # Force phone-level tokenization for CTC (not word-chunk strings like "bismi").
        phones = _norm_phone_str(
            res.phonemes_str(phoneme_sep=" ", word_sep=" ", verse_sep=" ")
        )
        surah_s, ayah_s = ref.split(":")
        rows.append(
            {
                "ID": f"{int(surah_s):03d}_{int(ayah_s):03d}",
                "quran_ref": ref,
                "surah": int(surah_s),
                "ayah": int(ayah_s),
                "wav_path": "",
                "duration_s": 0.0,
                "text": res.text().strip(),
                "phones_tajweed": phones,
                "phonemes": phones,
                "tajweed_match_score": float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(["surah", "ayah"]).reset_index(drop=True)


def split_train_dev(df: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(shuffled) * train_ratio)
    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    dev_df = shuffled.iloc[n_train:].reset_index(drop=True)
    return train_df, dev_df


def main() -> None:
    p = argparse.ArgumentParser(description="Build and split unified Tajweed metadata from sampled Quran refs.")
    p.add_argument("--output-dir", default="data/raw/tajweed_unified")
    p.add_argument("--save-full", action="store_true", help="Also save full unsplit metadata.csv.")
    p.add_argument("--verses-per-surah", type=int, default=3)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--audio-root",
        default=None,
        help="Optional local path to Quran-Data/public/audio for quran_ref->audio linking.",
    )
    p.add_argument("--audio-ext", default="m4a", help="Audio extension without dot (default: m4a).")
    args = p.parse_args()

    df = build_unified_metadata(
        verses_per_surah=args.verses_per_surah,
        seed=args.seed,
    )
    train_df, dev_df = split_train_dev(df=df, train_ratio=args.train_ratio, seed=args.seed)
    missing_train = 0
    missing_dev = 0
    if args.audio_root:
        train_df, missing_train = attach_audio(train_df, audio_root=args.audio_root, ext=args.audio_ext)
        dev_df, missing_dev = attach_audio(dev_df, audio_root=args.audio_root, ext=args.audio_ext)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / ("train_audio.csv" if args.audio_root else "train.csv")
    dev_path = out_dir / ("dev_audio.csv" if args.audio_root else "dev.csv")
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    if args.save_full:
        full_path = out_dir / "metadata.csv"
        df.to_csv(full_path, index=False)

    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved dev:   {dev_path} ({len(dev_df)} rows)")
    if args.audio_root:
        print(f"Audio linking: train missing={missing_train}, dev missing={missing_dev}")
    if args.save_full:
        print(f"Saved full:  {full_path} ({len(df)} rows)")
    print(
        f"Rows total: {len(df)} "
        f"(surahs=114, verses_per_surah={args.verses_per_surah}, train_ratio={args.train_ratio})"
    )
    print("Note: wav_path and duration_s are placeholders for text-only metadata.")


if __name__ == "__main__":
    main()
