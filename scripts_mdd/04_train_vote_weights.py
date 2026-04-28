#!/usr/bin/env python3
"""Train (expected, neighbor) k-NN vote weights on train; monitor dev; save JSON for 05.

For each **train** row where gold says the phone is correct but the weighted (or unweighted) vote
predicts *wrong* (false reject), we **down-weight** the pair (expected, majority_label) so that
neighbor label gets less total score on future votes.

**Weights file** (for ``05_retrieve_mdd.py --vote-weights``)::

  {
    "schema": "pair_vote_weights_v1",
    "default_weight": 1.0,
    "min_weight": 0.05,
    "pair_sep": "||",
    "pairs": { "x||y": 0.85, ... }
  }

Requires phoneme-level gold: ``phoneme_ref`` and ``phonemes`` in segment meta (same as 05).
Run 03 on ``unified/train`` and ``unified/dev`` first so embeddings align with those manifests.

**Neighbor cache:** FAISS search runs **once** per run (unless a valid cache is loaded). Top-``k``
bank indices for train/dev are written to ``knn_neighbors_k{K}.npz`` in ``--output-dir`` and reused
across all weight-tuning epochs (only the vote is recomputed).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

from retrieval_common import (
    REPO_ROOT,
    PAIR_SEP,
    as_abs,
    build_faiss_index,
    derive_phoneme_gold,
    load_embeddings_with_parts,
    pair_key,
    weighted_majority_label,
)

DEFAULT_BANK_EMB = REPO_ROOT / "data" / "unified" / "bank" / "embeddings" / "bank_embeddings.npy"
DEFAULT_BANK_META = REPO_ROOT / "data" / "unified" / "bank" / "embeddings" / "bank_embeddings_meta.csv"
DEFAULT_TRAIN_EMB = REPO_ROOT / "data" / "unified" / "train" / "embeddings" / "bank_embeddings.npy"
DEFAULT_TRAIN_META = REPO_ROOT / "data" / "unified" / "train" / "embeddings" / "bank_embeddings_meta.csv"
DEFAULT_DEV_EMB = REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings.npy"
DEFAULT_DEV_META = REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings_meta.csv"

NEIGHBORS_CACHE_SCHEMA = "knn_neighbors_cache"
RANDOM_SUBSAMPLE_SEED = 1337


def _subsample_rows_random(meta: pd.DataFrame, emb: np.ndarray, n_rows: int, seed: int, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly subsample aligned meta+embedding rows without replacement."""
    n_total = len(meta)
    if n_rows <= 0 or n_rows >= n_total:
        return meta, emb
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=n_rows, replace=False))
    print(f"[sample] {split}: random rows {n_rows}/{n_total} (seed={seed})")
    return meta.iloc[idx].reset_index(drop=True), emb[idx]


def _metrics_phoneme(gold_ok: np.ndarray, pred_ok: np.ndarray) -> dict[str, Any]:
    g = gold_ok.astype(bool)
    p = pred_ok.astype(bool)
    ta = int((g & p).sum())
    tr = int((~g & ~p).sum())
    fr = int((g & ~p).sum())
    fa = int((~g & p).sum())
    prec = tr / (tr + fr) if (tr + fr) else 0.0
    rec = tr / (tr + fa) if (tr + fa) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = float((g == p).mean()) if len(g) else 0.0
    return {
        "TA": ta,
        "TR": tr,
        "FR": fr,
        "FA": fa,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
    }


def _faiss_search_only(
    query: np.ndarray,
    index: object,
    k_eff: int,
    qbs: int,
    desc: str,
) -> np.ndarray:
    """Return int64 array [N, k_eff] of bank row indices (one-time FAISS)."""
    n = len(query)
    out = np.empty((n, k_eff), dtype=np.int64)
    n_batch = (n + qbs - 1) // qbs if qbs > 0 else 0
    for qstart in tqdm(
        range(0, n, qbs),
        desc=desc,
        total=n_batch,
        unit="batch",
        leave=True,
    ):
        qend = min(n, qstart + qbs)
        q_chunk = np.ascontiguousarray(query[qstart:qend])
        _, idx_chunk = index.search(q_chunk, k_eff)
        out[qstart:qend] = idx_chunk
    return out


def _predict_maj_from_cache(
    qm: pd.DataFrame,
    nbr_idx: np.ndarray,
    bank_lab: np.ndarray,
    k_eff: int,
    pairs: dict[str, float],
    default_w: float,
    row_chunk: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return majority_label (object array), pred_is_correct (bool) from precomputed top-k bank indices."""
    n = len(qm)
    if nbr_idx.shape[0] != n or nbr_idx.shape[1] != k_eff:
        raise ValueError(
            f"nbr_idx shape {nbr_idx.shape} != rows={n} k_eff={k_eff}"
        )
    maj = np.empty(n, dtype=object)
    pred = np.zeros(n, dtype=bool)
    rc = max(1, int(row_chunk))
    for bstart in tqdm(
        range(0, n, rc),
        desc=desc,
        total=(n + rc - 1) // rc,
        unit="chunk",
        leave=False,
    ):
        bend = min(n, bstart + rc)
        for qi in range(bstart, bend):
            expected = str(qm.at[qi, "expected_phoneme"])
            nbr = [str(bank_lab[int(j)]) for j in nbr_idx[qi]]
            mlab = weighted_majority_label(nbr, expected, pairs, default_w)
            maj[qi] = mlab
            pred[qi] = mlab == expected
    return maj, pred


def _neighbors_cache_paths(out_dir: Path, k_eff: int) -> tuple[Path, Path]:
    return out_dir / f"knn_neighbors_k{k_eff}.npz", out_dir / f"knn_neighbors_k{k_eff}_meta.json"


def _neighbors_cache_loadable(
    npz_path: Path,
    meta_path: Path,
    k_eff: int,
    n_train: int,
    n_dev: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema") != NEIGHBORS_CACHE_SCHEMA:
            return None
        if int(meta.get("k_eff", -1)) != k_eff:
            return None
        d = np.load(npz_path)
        tr = d["train_nbr_idx"]
        dv = d["dev_nbr_idx"]
    except Exception:
        return None
    if tr.shape != (n_train, k_eff) or dv.shape != (n_dev, k_eff):
        print(
            f"[cache] shape mismatch: have {tr.shape}, {dv.shape} "
            f"need ({n_train},{k_eff}), ({n_dev},{k_eff}) — will recompute FAISS."
        )
        return None
    return tr, dv


def _neighbors_cache_save(
    npz_path: Path,
    meta_path: Path,
    train_nbr: np.ndarray,
    dev_nbr: np.ndarray,
    k_eff: int,
    n_bank: int,
    dim: int,
    path_info: dict[str, str],
) -> None:
    np.savez_compressed(npz_path, train_nbr_idx=train_nbr, dev_nbr_idx=dev_nbr)
    meta = {
        "schema": NEIGHBORS_CACHE_SCHEMA,
        "k_eff": k_eff,
        "n_bank": n_bank,
        "dim": dim,
        "n_train": int(train_nbr.shape[0]),
        "n_dev": int(dev_nbr.shape[0]),
        "paths": path_info,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[cache] wrote {npz_path} and {meta_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train k-NN pair vote weights on train, tune on dev metrics.")
    p.add_argument("--bank-embeddings", default=str(DEFAULT_BANK_EMB), help="Bank .npy from 03 (correct-only).")
    p.add_argument("--bank-meta", default=str(DEFAULT_BANK_META))
    p.add_argument("--train-query-embeddings", default=str(DEFAULT_TRAIN_EMB), help="Train query .npy from 03.")
    p.add_argument("--train-query-meta", default=str(DEFAULT_TRAIN_META))
    p.add_argument("--dev-query-embeddings", default=str(DEFAULT_DEV_EMB), help="Dev query .npy from 03.")
    p.add_argument("--dev-query-meta", default=str(DEFAULT_DEV_META))
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: data/unified/train/vote_train_{top_k} under repo root.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20, help="Passes over train-based FR pair updates.")
    p.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Subtracted from w(x,y) for each (expected=x, majority=y) on train FR rows (after grouping counts).",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Floor for each pair weight w(x,y).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Always-on early stopping patience on dev F1 (epochs without >= min-delta improvement).",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum dev F1 increase to qualify as improvement.",
    )
    p.add_argument(
        "--faiss-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    p.add_argument("--faiss-gpu-id", type=int, default=0)
    p.add_argument(
        "--faiss-query-batch-size",
        type=int,
        default=16,
        help="Query batch size for the one-time FAISS search (not used per epoch).",
    )
    p.add_argument(
        "--neighbors-cache-dir",
        default=None,
        help="Directory for knn_neighbors_k{K}.npz (default: same as --output-dir).",
    )
    p.add_argument(
        "--recompute-neighbors",
        action="store_true",
        help="Re-run FAISS and overwrite the neighbor index cache.",
    )
    p.add_argument(
        "--vote-predict-chunk",
        type=int,
        default=16384,
        help="Query-meta rows per tqdm chunk when only recomputing votes (per epoch).",
    )
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=-1,
        help="If >0, cap train rows (debug).",
    )
    p.add_argument(
        "--max-dev-rows",
        type=int,
        default=-1,
        help="If >0, cap dev rows (debug).",
    )
    args = p.parse_args()
    k = int(args.top_k)
    if k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.output_dir is None:
        out_dir = REPO_ROOT / "data" / "unified" / "train" / f"vote_train_{k}"
    else:
        out_dir = as_abs(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = as_abs(args.neighbors_cache_dir) if args.neighbors_cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss  # type: ignore  # noqa: F401
    except Exception as e:
        raise SystemExit("FAISS is required. Install faiss-cpu (or faiss-gpu).") from e

    t0 = time.perf_counter()
    print("[startup] 04_train_vote_weights.py")

    bank, bank_m, _ = load_embeddings_with_parts(as_abs(args.bank_embeddings), as_abs(args.bank_meta), "bank")
    q_tr, m_tr, _ = load_embeddings_with_parts(
        as_abs(args.train_query_embeddings), as_abs(args.train_query_meta), "train"
    )
    q_dv, m_dv, _ = load_embeddings_with_parts(
        as_abs(args.dev_query_embeddings), as_abs(args.dev_query_meta), "dev"
    )
    if args.max_train_rows and args.max_train_rows > 0:
        m_tr, q_tr = _subsample_rows_random(
            m_tr, q_tr, int(args.max_train_rows), RANDOM_SUBSAMPLE_SEED, split="train"
        )
    if args.max_dev_rows and args.max_dev_rows > 0:
        m_dv, q_dv = _subsample_rows_random(
            m_dv, q_dv, int(args.max_dev_rows), RANDOM_SUBSAMPLE_SEED + 1, split="dev"
        )

    for col in ("expected_phoneme", "utt_id"):
        if col not in m_tr.columns or col not in m_dv.columns:
            raise SystemExit(f"train and dev query meta need column: {col}")

    m_tr = derive_phoneme_gold(m_tr)
    m_dv = derive_phoneme_gold(m_dv)
    if "gold_is_phoneme_correct" not in m_tr.columns or m_tr["gold_is_phoneme_correct"].isna().all():
        raise SystemExit(
            "Phoneme-level gold missing on train. Ensure segment CSV (03 meta) has phoneme_ref and phonemes."
        )
    m_tr["gold_is_phoneme_correct"] = m_tr["gold_is_phoneme_correct"].fillna(False).astype(bool)
    m_dv["gold_is_phoneme_correct"] = m_dv["gold_is_phoneme_correct"].fillna(False).astype(bool)
    g_tr = m_tr["gold_is_phoneme_correct"].values.astype(bool)
    g_dv = m_dv["gold_is_phoneme_correct"].values.astype(bool)

    bank = bank.astype(np.float32, copy=False)
    q_tr = q_tr.astype(np.float32, copy=False)
    q_dv = q_dv.astype(np.float32, copy=False)
    if q_tr.shape[1] != bank.shape[1] or q_dv.shape[1] != bank.shape[1]:
        raise SystemExit("Embedding dim mismatch between bank and query.")

    k_eff = min(k, len(bank))
    index, faiss_dev, n_gpu = build_faiss_index(bank, str(args.faiss_device), int(args.faiss_gpu_id))
    index.add(np.ascontiguousarray(bank))
    bank_lab = bank_m["expected_phoneme"].astype(str).values
    print(f"[faiss] used_device={faiss_dev} n_gpus={n_gpu} k_eff={k_eff}")

    qbs = max(1, int(args.faiss_query_batch_size))
    vote_chunk = max(1, int(args.vote_predict_chunk))

    npz_path, meta_path = _neighbors_cache_paths(cache_dir, k_eff)
    cached = _neighbors_cache_loadable(
        npz_path, meta_path, k_eff, len(m_tr), len(m_dv)
    )
    loaded_cache = cached is not None and not args.recompute_neighbors
    if loaded_cache:
        assert cached is not None
        train_nbr_idx, dev_nbr_idx = cached
        print(
            f"[cache] loaded neighbor indices: {npz_path} "
            f"train{train_nbr_idx.shape} dev{dev_nbr_idx.shape}"
        )
    else:
        if args.recompute_neighbors:
            print("[cache] --recompute-neighbors: running one-time FAISS search")
        else:
            print("[cache] no valid neighbor cache; running one-time FAISS search")
        train_nbr_idx = _faiss_search_only(q_tr, index, k_eff, qbs, "FAISS train (once)")
        dev_nbr_idx = _faiss_search_only(q_dv, index, k_eff, qbs, "FAISS dev (once)")
        _neighbors_cache_save(
            npz_path,
            meta_path,
            train_nbr_idx,
            dev_nbr_idx,
            k_eff,
            int(len(bank)),
            int(bank.shape[1]),
            {
                "bank_embeddings": str(as_abs(args.bank_embeddings).resolve()),
                "train_query": str(as_abs(args.train_query_embeddings).resolve()),
                "dev_query": str(as_abs(args.dev_query_embeddings).resolve()),
            },
        )
    # Drop FAISS index; remaining epochs only use bank labels + cached indices.
    del index
    del bank, q_tr, q_dv
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    pairs: dict[str, float] = {}
    default_w = 1.0
    wmin = float(args.min_weight)
    step = float(args.step)
    epochs = int(args.epochs)

    log: list[dict[str, Any]] = []
    best_dev_f1 = -1.0
    best_pairs: dict[str, float] | None = None
    last_m_dev: dict[str, Any] = {}
    best_epoch = -1
    epochs_no_improve = 0
    last_epoch_idx = -1
    patience = max(1, int(args.patience))
    min_delta = max(0.0, float(args.min_delta))

    for ep in range(epochs):
        last_epoch_idx = ep
        t_ep = time.perf_counter()
        maj_tr, pred_tr = _predict_maj_from_cache(
            m_tr,
            train_nbr_idx,
            bank_lab,
            k_eff,
            pairs,
            default_w,
            vote_chunk,
            desc=f"epoch {ep} train (vote from cache)",
        )
        _, pred_dv = _predict_maj_from_cache(
            m_dv,
            dev_nbr_idx,
            bank_lab,
            k_eff,
            pairs,
            default_w,
            vote_chunk,
            desc=f"epoch {ep} dev (vote from cache)",
        )
        m_train = _metrics_phoneme(g_tr, pred_tr)
        m_dev = _metrics_phoneme(g_dv, pred_dv)
        last_m_dev = m_dev
        log.append(
            {
                "epoch": ep,
                "train": m_train,
                "dev": m_dev,
                "n_stored_pairs": len(pairs),
                "sec": time.perf_counter() - t_ep,
                "epochs_no_improve": epochs_no_improve,
            }
        )
        print(
            f"[epoch {ep}] train f1={m_train['f1_score']:.4f} acc={m_train['accuracy']:.4f} | "
            f"dev f1={m_dev['f1_score']:.4f} acc={m_dev['accuracy']:.4f} | n_pairs={len(pairs)}"
        )

        improved = (m_dev["f1_score"] - best_dev_f1) >= min_delta
        if improved:
            best_dev_f1 = m_dev["f1_score"]
            best_pairs = pairs.copy()
            best_epoch = ep
            epochs_no_improve = 0
            p_best = out_dir / "vote_weights.best_dev.json"
            _save_weights(p_best, best_pairs, default_w, wmin, k, m_dev, tag="best_dev_f1", ep=ep)
            print(f"  [dev] new best f1 (min_delta={min_delta:g}) -> {p_best}")
        else:
            epochs_no_improve += 1
            print(
                f"  [dev] no improvement (best={best_dev_f1:.4f}, "
                f"delta={m_dev['f1_score'] - best_dev_f1:+.6f}); "
                f"patience {epochs_no_improve}/{patience}"
            )
            if epochs_no_improve >= patience:
                print("  [dev] early stopping triggered.")
                break

        if ep == epochs - 1:
            break

        # Train FR: down-weight w(expected, majority_label) that led to false reject
        fr_idx = np.where(g_tr & (~pred_tr))[0]
        if len(fr_idx) == 0:
            print("  [train] no FR rows; stopping early.")
            break
        ctr: Counter[str] = Counter()
        for qi in fr_idx:
            x = str(m_tr.at[qi, "expected_phoneme"])
            y = str(maj_tr[qi])
            ctr[pair_key(x, y)] += 1
        for key, c in ctr.items():
            cur = pairs.get(key, default_w)
            pairs[key] = max(wmin, cur - step * float(c))
        print(f"  [train] FR={len(fr_idx)} unique_pairs={len(ctr)}; applied weight nudges")

    # Final save (weights after last train update, or after last eval if early exit)
    final_path = out_dir / "vote_weights.json"
    _save_weights(
        final_path, pairs, default_w, wmin, k, last_m_dev, tag="last_epoch", ep=last_epoch_idx
    )
    log_path = out_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "neighbors_cache": {
                    "npz": str(npz_path),
                    "meta": str(meta_path),
                    "loaded_from_disk": bool(loaded_cache),
                },
                "best_dev_f1": best_dev_f1,
                "best_epoch": best_epoch,
                "early_stop": {
                    "enabled": True,
                    "patience": patience,
                    "min_delta": min_delta,
                    "epochs_no_improve_at_end": epochs_no_improve,
                    "stopped_early": bool(last_epoch_idx + 1 < epochs),
                },
                "epochs": epochs,
                "pair_sep": PAIR_SEP,
                "per_epoch": log,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {final_path}")
    if best_pairs is not None:
        print(f"Best dev f1={best_dev_f1:.4f} (see vote_weights.best_dev.json)")
    print(f"Wrote {log_path}")
    print(f"[timing] total_elapsed_sec={time.perf_counter() - t0:.2f}")
    return 0


def _save_weights(
    path: Path,
    pairs: dict[str, float],
    default_w: float,
    wmin: float,
    top_k: int,
    metrics: dict[str, Any],
    *,
    tag: str,
    ep: int = -1,
) -> None:
    out = {
        "schema": "pair_vote_weights_v1",
        "tag": tag,
        "epoch": ep,
        "pair_sep": PAIR_SEP,
        "default_weight": default_w,
        "min_weight": wmin,
        "top_k": top_k,
        "pairs": {k: float(v) for k, v in sorted(pairs.items(), key=lambda kv: kv[0])},
        "metrics_snapshot": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())