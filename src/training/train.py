"""Step 2 trainer: frozen SSL + BiLSTM CTC head."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from typing import Dict, List

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import build_dataloader
from src.data.vocab import decode_ids, load_vocab
from src.evaluation.metrics import evaluate_phoneme_recognition
from src.models.ctc_model import FrozenSSLCTC
from src.utils.decoding import decode_ids_to_phones, greedy_decode_ids


def _batch_to_device(batch: Dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _collect_refs(batch, vocab) -> List[List[str]]:
    refs = []
    labels = batch["labels"].detach().cpu()
    lens = batch["label_len"].detach().cpu().tolist()
    for i, ln in enumerate(lens):
        ids = labels[i, :ln].tolist()
        refs.append(decode_ids(ids, vocab).split())
    return refs


@torch.no_grad()
def evaluate(model, loader, vocab, blank_id, device):
    model.eval()
    all_refs: List[List[str]] = []
    all_preds: List[List[str]] = []
    losses = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = _batch_to_device(batch, device)
        out = model(
            audio=batch["audio"],
            audio_len=batch["audio_len"],
            labels=batch["labels"],
            label_len=batch["label_len"],
        )
        losses.append(float(out["loss"].item()))
        pred_ids = greedy_decode_ids(out["logits"], blank_id=blank_id)
        pred_phones = decode_ids_to_phones(pred_ids, vocab)
        refs = _collect_refs(batch, vocab)
        all_refs.extend(refs)
        all_preds.extend(pred_phones)

    metrics = evaluate_phoneme_recognition(all_refs, all_preds)
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 4)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/step2_frozen_xls_r_300m.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using config: {args.config}")

    vocab = load_vocab(cfg["vocab_path"])
    blank_id = int(vocab["<blank>"])

    train_loader = build_dataloader(
        csv_path=cfg["train_csv"],
        vocab=vocab,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        max_audio_len_sec=float(cfg["max_audio_len_sec"]),
    )
    val_loader = build_dataloader(
        csv_path=cfg["val_csv"],
        vocab=vocab,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        max_audio_len_sec=float(cfg["max_audio_len_sec"]),
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

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    best_per = 1e9
    patience = int(cfg.get("patience", 3))
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        running = 0.0

        for batch in pbar:
            batch = _batch_to_device(batch, device)
            out = model(
                audio=batch["audio"],
                audio_len=batch["audio_len"],
                labels=batch["labels"],
                label_len=batch["label_len"],
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optim.step()
            optim.zero_grad(set_to_none=True)

            running += float(loss.item())
            global_step += 1
            if global_step % int(cfg["log_every_steps"]) == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
            pbar.set_postfix({"loss": f"{(running/max(1,global_step)):.4f}"})

        metrics = evaluate(model, val_loader, vocab, blank_id, device)
        print(f"Epoch {epoch} val metrics: {metrics}")
        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/PER", metrics["PER"], epoch)
        writer.add_scalar("val/Accuracy", metrics["Accuracy"], epoch)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "metrics": metrics,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if metrics["PER"] < best_per:
            best_per = metrics["PER"]
            bad_epochs = 0
            torch.save(ckpt, output_dir / "best.pt")
            with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"New best checkpoint (PER={best_per:.4f})")
        else:
            bad_epochs += 1
            print(f"No improvement for {bad_epochs}/{patience} epoch(s)")

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best PER={best_per:.4f})")
            break

    writer.close()
    print(f"Training done. Best PER={best_per:.4f}")


if __name__ == "__main__":
    main()

