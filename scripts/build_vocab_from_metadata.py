"""Build phoneme vocab JSON from metadata CSV phoneme column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SPECIAL_TOKENS = ["<pad>", "<unk>", "<blank>"]


def build_vocab_from_csv(
    input_csv: str,
    output_json: str,
    phoneme_col: str = "phonemes",
) -> dict[str, int]:
    df = pd.read_csv(input_csv)
    if phoneme_col not in df.columns:
        raise ValueError(f"Column {phoneme_col!r} not found in {input_csv}")

    token_set: set[str] = set()
    for s in df[phoneme_col].dropna().astype(str):
        for tok in s.strip().split():
            if tok and tok != "<sil>":
                token_set.add(tok)

    tokens = sorted(token_set)
    vocab: dict[str, int] = {}
    for i, t in enumerate(SPECIAL_TOKENS):
        vocab[t] = i
    for i, t in enumerate(tokens, start=len(SPECIAL_TOKENS)):
        vocab[t] = i

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return vocab


def main() -> None:
    p = argparse.ArgumentParser(description="Build vocab JSON from metadata phoneme labels.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--phoneme-col", default="phonemes")
    args = p.parse_args()

    vocab = build_vocab_from_csv(
        input_csv=args.input_csv,
        output_json=args.output_json,
        phoneme_col=args.phoneme_col,
    )
    print(f"Saved vocab: {args.output_json}")
    print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
