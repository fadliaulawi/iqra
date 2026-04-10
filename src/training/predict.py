"""Inference script for Step 2 checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data.dataset import build_dataloader
from src.data.vocab import load_vocab
from src.models.ctc_model import FrozenSSLCTC
from src.utils.decoding import decode_ids_to_phones, greedy_decode_ids


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/step2_frozen_mms_1b_all.yaml")
    p.add_argument("--checkpoint", default="outputs/step2_frozen_mms_1b_all/best.pt")
    p.add_argument("--input-csv", default="data/raw/test/metadata.csv")
    p.add_argument("--output-csv", default="outputs/step2_frozen_mms_1b_all/predictions.csv")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vocab = load_vocab(cfg["vocab_path"])
    blank_id = int(vocab["<blank>"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_dataloader(
        csv_path=args.input_csv,
        vocab=vocab,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        max_audio_len_sec=float(cfg["max_audio_len_sec"]),
        phoneme_col=None,  # inference mode
    )

    model = FrozenSSLCTC(
        encoder_name=cfg["encoder_name"],
        num_labels=len(vocab),
        blank_id=blank_id,
        head_hidden=int(cfg["head_hidden"]),
        head_layers=int(cfg["head_layers"]),
        head_dropout=float(cfg["head_dropout"]),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        mms_target_lang=cfg.get("mms_target_lang") or None,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows = []
    for batch in tqdm(loader, desc="Predict"):
        batch = _batch_to_device(batch, device)
        out = model(audio=batch["audio"], audio_len=batch["audio_len"])
        pred_ids = greedy_decode_ids(out["logits"], blank_id=blank_id)
        pred_phones = decode_ids_to_phones(pred_ids, vocab)
        for uid, phones in zip(batch["ids"], pred_phones):
            rows.append({"ID": uid, "Prediction": " ".join(phones)})

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()

