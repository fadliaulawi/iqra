"""Step 2/3 trainer: SSL or Whisper encoder + BiLSTM CTC head."""

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
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset import build_dataloader
from src.data.vocab import decode_ids, load_vocab
from src.evaluation.metrics import evaluate_phoneme_recognition
from src.models.ctc_model import FrozenSSLCTC
from src.models.whisper_phoneme import WhisperEncoderCTC
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
    p.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint (.pt). Restores full training state unless --reset-optimizer is used.",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Alias of --resume (useful for multi-stage scripts).",
    )
    p.add_argument(
        "--stage",
        default=None,
        help="Optional stage label for logging (e.g., S1, S2, S3).",
    )
    p.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights from resume checkpoint but start a fresh optimizer/scheduler/scaler state.",
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using config: {args.config}")
    if args.stage:
        print(f"Stage: {args.stage}")

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

    model_type = cfg.get("model_type", "ssl_ctc")
    common_kwargs = dict(
        encoder_name=cfg["encoder_name"],
        num_labels=len(vocab),
        blank_id=blank_id,
        head_hidden=int(cfg["head_hidden"]),
        head_layers=int(cfg["head_layers"]),
        head_dropout=float(cfg["head_dropout"]),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        specaugment=bool(cfg.get("specaugment", False)),
    )

    if model_type == "whisper_encoder_ctc":
        model = WhisperEncoderCTC(**common_kwargs).to(device)
    else:
        model = FrozenSSLCTC(
            **common_kwargs,
            mms_target_lang=cfg.get("mms_target_lang") or None,
        ).to(device)

    # Split optimizer groups by membership in `model.encoder`, not name prefix
    encoder_param_set = set(model.encoder.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_seen = set()
    head_params = []
    for _, p in model.named_parameters():
        if not p.requires_grad or p in encoder_param_set:
            continue
        pid = id(p)
        if pid in head_seen:
            continue
        head_seen.add(pid)
        head_params.append(p)

    if bool(cfg.get("freeze_encoder", True)):
        optim = torch.optim.AdamW(
            head_params,
            lr=float(cfg.get("learning_rate", 3e-4)),
            weight_decay=float(cfg["weight_decay"]),
        )
    else:
        optim = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": float(cfg.get("encoder_lr", 1e-5))},
                {"params": head_params, "lr": float(cfg.get("head_lr", 3e-4))},
            ],
            weight_decay=float(cfg["weight_decay"]),
        )

    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    total_steps = (len(train_loader) * int(cfg["max_epochs"])) // max(1, grad_accum)
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )
    use_fp16 = bool(cfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    resume_path = args.resume_from or args.resume or cfg.get("resume_from")
    start_epoch = 0
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    best_per = 1e9
    patience = int(cfg.get("patience", 3))
    bad_epochs = 0
    global_step = 0

    if resume_path:
        ckpt_path = Path(resume_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if args.reset_optimizer:
            print(
                f"Loaded weights from {ckpt_path}; starting fresh optimizer/scheduler state "
                f"(stage transfer mode)."
            )
        else:
            optim.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            if "best_per" in ckpt:
                best_per = float(ckpt["best_per"])
            else:
                sidecar = output_dir / "best_metrics.json"
                if sidecar.is_file():
                    with open(sidecar, encoding="utf-8") as f:
                        best_per = float(json.load(f)["PER"])
                else:
                    best_per = float(ckpt.get("metrics", {}).get("PER", 1e9))
                    print("Warning: legacy checkpoint without best_per; approximated best PER from saved metrics.")
            bad_epochs = int(ckpt.get("bad_epochs", 0))
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            else:
                print(
                    "Warning: checkpoint has no scheduler state; LR schedule restarts from current optimizer step. "
                    "Prefer resuming from checkpoints saved after this trainer update."
                )
            if use_fp16 and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            print(
                f"Resumed from {ckpt_path} (completed epoch {start_epoch}, global_step={global_step}, best_per={best_per:.4f})"
            )

    max_epochs = int(cfg["max_epochs"])
    if start_epoch >= max_epochs:
        print(
            f"Nothing to train: resume epoch {start_epoch} >= max_epochs {max_epochs}. "
            f"Increase max_epochs in the config and run again with --resume."
        )
        writer.close()
        return

    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        running = 0.0

        optim.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(pbar, start=1):
            batch = _batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(
                    audio=batch["audio"],
                    audio_len=batch["audio_len"],
                    labels=batch["labels"],
                    label_len=batch["label_len"],
                )
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()
            if step_idx % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            running += float((loss * grad_accum).item())
            global_step += 1
            if global_step % int(cfg["log_every_steps"]) == 0:
                writer.add_scalar("train/loss", float((loss * grad_accum).item()), global_step)
            pbar.set_postfix({"loss": f"{(running/max(1,global_step)):.4f}"})

        # Flush remaining gradients when batches are not divisible by accumulation steps.
        if len(train_loader) % grad_accum != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

        metrics = evaluate(model, val_loader, vocab, blank_id, device)
        print(f"Epoch {epoch} val metrics: {metrics}")
        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/PER", metrics["PER"], epoch)
        writer.add_scalar("val/Accuracy", metrics["Accuracy"], epoch)

        if metrics["PER"] < best_per:
            best_per = metrics["PER"]
            bad_epochs = 0
            print(f"New best checkpoint (PER={best_per:.4f})")
        else:
            bad_epochs += 1
            print(f"No improvement for {bad_epochs}/{patience} epoch(s)")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_per": best_per,
            "bad_epochs": bad_epochs,
            "config": cfg,
            "metrics": metrics,
        }
        if use_fp16:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, output_dir / "last.pt")

        if bad_epochs == 0:
            torch.save(ckpt, output_dir / "best.pt")
            with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best PER={best_per:.4f})")
            break

    writer.close()
    print(f"Training done. Best PER={best_per:.4f}")


if __name__ == "__main__":
    main()

