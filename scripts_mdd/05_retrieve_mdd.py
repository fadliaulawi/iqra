#!/usr/bin/env python3
"""Retrieval-based MDD on precomputed query and bank embeddings (inference, plan 6–8).

Run 03 to build bank + query embeddings, then this script. Optional JSON pair weights
(from 04_train_vote_weights.py) reweight k-NN votes per (expected_phoneme, neighbor label).

Default output: ``data/unified/dev/retrieval_{TOP_K}/`` (override with ``--output-dir``).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from retrieval_common import (
    REPO_ROOT,
    PAIR_SEP,
    as_abs,
    build_faiss_index,
    crosstab_expected_vs_majority,
    derive_phoneme_gold,
    load_embeddings_with_parts,
    load_vote_weights_json,
    weighted_majority_label,
)

DEFAULT_BANK_EMB = REPO_ROOT / "data" / "unified" / "bank" / "embeddings" / "bank_embeddings.npy"
DEFAULT_BANK_META = REPO_ROOT / "data" / "unified" / "bank" / "embeddings" / "bank_embeddings_meta.csv"
DEFAULT_QUERY_EMB = REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings.npy"
DEFAULT_QUERY_META = REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings_meta.csv"
DEFAULT_QUERY_BY_SET = {
    "dev": {
        "embeddings": REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings.npy",
        "meta": REPO_ROOT / "data" / "unified" / "dev" / "embeddings" / "bank_embeddings_meta.csv",
    },
    "test": {
        "embeddings": REPO_ROOT / "data" / "unified" / "test" / "embeddings" / "bank_embeddings.npy",
        "meta": REPO_ROOT / "data" / "unified" / "test" / "embeddings" / "bank_embeddings_meta.csv",
    },
}
NEIGHBORS_CACHE_SCHEMA = "knn_neighbors_cache_v1"


def _neighbors_cache_paths(cache_dir: Path, k_eff: int) -> tuple[Path, Path]:
    return cache_dir / f"knn_neighbors_k{k_eff}.npz", cache_dir / f"knn_neighbors_k{k_eff}_meta.json"


def _neighbors_cache_loadable(
    npz_path: Path,
    meta_path: Path,
    *,
    k_eff: int,
    n_query: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema") != NEIGHBORS_CACHE_SCHEMA:
            return None
        if int(meta.get("k_eff", -1)) != int(k_eff):
            return None
        d = np.load(npz_path)
        idx = d["query_nbr_idx"]
        sims = d["query_nbr_sims"]
    except Exception:
        return None
    if idx.shape != (n_query, k_eff) or sims.shape != (n_query, k_eff):
        return None
    return idx, sims


def _neighbors_cache_save(
    npz_path: Path,
    meta_path: Path,
    *,
    idx: np.ndarray,
    sims: np.ndarray,
    k_eff: int,
    n_bank: int,
    dim: int,
    path_info: dict[str, str],
) -> None:
    np.savez_compressed(npz_path, query_nbr_idx=idx, query_nbr_sims=sims)
    meta = {
        "schema": NEIGHBORS_CACHE_SCHEMA,
        "k_eff": int(k_eff),
        "n_query": int(idx.shape[0]),
        "n_bank": int(n_bank),
        "dim": int(dim),
        "paths": path_info,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[cache] wrote {npz_path} and {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="k-NN retrieval MDD (bank vs query embeddings).")
    parser.add_argument(
        "--dataset-set",
        choices=["dev", "test"],
        default="dev",
        help="Preset for query embeddings/meta and default output dir.",
    )
    parser.add_argument("--bank-embeddings", default=str(DEFAULT_BANK_EMB), help="Path to bank .npy [N, D].")
    parser.add_argument("--bank-meta", default=str(DEFAULT_BANK_META), help="bank_embeddings_meta.csv")
    parser.add_argument(
        "--query-embeddings",
        default=None,
        help="Path to query .npy from 03 (same D as bank, L2-normed).",
    )
    parser.add_argument(
        "--query-meta",
        default=None,
        help="CSV with same row order as query embeddings: must include expected_phoneme, utt_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/unified/dev/retrieval_{TOP_K} under repo root).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of bank neighbors to vote over.")
    parser.add_argument(
        "--vote-weights",
        default=None,
        help="Optional JSON from 04_train_vote_weights.py (pair weights for neighbor votes).",
    )
    parser.add_argument(
        "--faiss-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="FAISS execution device. auto prefers GPU when available.",
    )
    parser.add_argument(
        "--faiss-gpu-id",
        type=int,
        default=0,
        help="GPU id used when --faiss-device is gpu/auto with available GPUs.",
    )
    parser.add_argument(
        "--faiss-query-batch-size",
        type=int,
        default=16,
        help="Number of query vectors per FAISS search call (memory/speed tradeoff).",
    )
    parser.add_argument(
        "--neighbors-cache-dir",
        default=None,
        help="Directory for knn_neighbors_k{K}.npz (default: same as --output-dir).",
    )
    parser.add_argument(
        "--recompute-neighbors",
        action="store_true",
        help="Ignore cache and rerun FAISS search, then overwrite cache.",
    )
    parser.add_argument(
        "--max-query-rows",
        type=int,
        default=-1,
        help="If >0, only first N query rows (debug).",
    )
    args = parser.parse_args()
    k = int(args.top_k)
    if k < 1:
        raise SystemExit("--top-k must be >= 1")
    query_defaults = DEFAULT_QUERY_BY_SET[str(args.dataset_set)]
    if args.output_dir is None:
        out_dir = REPO_ROOT / "data" / "unified" / str(args.dataset_set) / f"retrieval_{k}"
    else:
        out_dir = as_abs(args.output_dir)
    t0 = time.perf_counter()
    print("[startup] 05_retrieve_mdd.py started")

    try:
        import faiss  # type: ignore  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "FAISS is required. Install faiss-cpu (or faiss-gpu) in your environment."
        ) from e

    vote_pairs: dict[str, float] = {}
    vote_default = 1.0
    if args.vote_weights:
        wpath = as_abs(args.vote_weights)
        if not wpath.is_file():
            raise SystemExit(f"--vote-weights not found: {wpath}")
        vote_pairs, vote_default = load_vote_weights_json(wpath)
        print(f"[startup] loaded vote weights: n_pairs={len(vote_pairs)} default={vote_default} path={wpath}")
    else:
        print("[startup] unweighted k-NN vote (all neighbor weights=1)")

    bank_p = as_abs(args.bank_embeddings)
    bank_meta_p = as_abs(args.bank_meta)
    qemb_p = as_abs(args.query_embeddings) if args.query_embeddings else Path(query_defaults["embeddings"])
    qmeta_p = as_abs(args.query_meta) if args.query_meta else Path(query_defaults["meta"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = as_abs(args.neighbors_cache_dir) if args.neighbors_cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[startup] output_dir={out_dir}")
    out_csv = out_dir / "mdd_retrieval_results.csv"
    out_json = out_dir / "mdd_retrieval_stats.json"

    t_bank_load = time.perf_counter()
    bank, bank_m, bank_load_info = load_embeddings_with_parts(bank_p, bank_meta_p, role="bank")
    print(
        f"[timing] loaded bank embeddings/meta in {time.perf_counter() - t_bank_load:.2f}s "
        f"(rows={len(bank)})"
    )
    if bank.ndim != 2:
        raise SystemExit("bank_embeddings must be 2D [N, D]")
    t_query_load = time.perf_counter()
    query, qm, query_load_info = load_embeddings_with_parts(qemb_p, qmeta_p, role="query")
    print(
        f"[timing] loaded query embeddings/meta in {time.perf_counter() - t_query_load:.2f}s "
        f"(rows={len(query)})"
    )
    if query.ndim != 2:
        raise SystemExit("query_embeddings must be 2D [Q, D]")
    if query.shape[1] != bank.shape[1]:
        raise SystemExit(
            f"Dim mismatch: bank D={bank.shape[1]} vs query D={query.shape[1]}"
        )
    if args.max_query_rows and args.max_query_rows > 0:
        qm = qm.head(args.max_query_rows).copy()
        query = query[: len(qm)]

    if "expected_phoneme" not in qm.columns or "utt_id" not in qm.columns:
        raise SystemExit("query_meta must include utt_id and expected_phoneme")
    if "expected_phoneme" not in bank_m.columns or "utt_id" not in bank_m.columns:
        raise SystemExit("bank_meta must include utt_id and expected_phoneme")
    if len(bank_m) != len(bank):
        raise SystemExit("bank_meta rows must match bank_embeddings length")
    if len(qm) != len(query):
        raise SystemExit("query_meta rows must match query_embeddings length")

    t_gold = time.perf_counter()
    qm = derive_phoneme_gold(qm)
    print(f"[timing] derived phoneme-level gold in {time.perf_counter() - t_gold:.2f}s")

    bank = bank.astype(np.float32, copy=False)
    query = query.astype(np.float32, copy=False)
    if len(bank) == 0:
        raise SystemExit("Empty bank embeddings.")
    k_eff = min(k, int(len(bank)))
    t_index_create = time.perf_counter()
    index, faiss_device_used, faiss_num_gpus = build_faiss_index(
        bank, str(args.faiss_device), int(args.faiss_gpu_id)
    )
    t_index_create_s = time.perf_counter() - t_index_create
    print(f"[timing] faiss index build in {t_index_create_s:.2f}s")

    t_index_add = time.perf_counter()
    index.add(np.ascontiguousarray(bank))
    t_index_add_s = time.perf_counter() - t_index_add
    print(f"[timing] faiss index.add(bank) in {t_index_add_s:.2f}s")
    print(
        f"[faiss] requested_device={args.faiss_device} used_device={faiss_device_used} "
        f"faiss_num_gpus={faiss_num_gpus} index_type={type(index).__name__} "
        f"n_bank={len(bank)} dim={bank.shape[1]} top_k={k_eff}"
    )
    bank_lab = bank_m["expected_phoneme"].astype(str).values

    qbs = max(1, int(args.faiss_query_batch_size))
    t_retrieval_start = time.perf_counter()
    npz_path, meta_path = _neighbors_cache_paths(cache_dir, k_eff)
    cached = _neighbors_cache_loadable(npz_path, meta_path, k_eff=k_eff, n_query=len(qm))
    loaded_cache = cached is not None and not args.recompute_neighbors
    if loaded_cache:
        assert cached is not None
        all_idx, all_sims = cached
        print(f"[cache] loaded neighbor cache: {npz_path} shape={all_idx.shape}")
    else:
        if args.recompute_neighbors:
            print("[cache] --recompute-neighbors set; running FAISS search")
        else:
            print("[cache] no valid cache; running FAISS search")
        all_idx = np.empty((len(qm), k_eff), dtype=np.int64)
        all_sims = np.empty((len(qm), k_eff), dtype=np.float32)
        for qstart in tqdm(range(0, len(qm), qbs), desc="Retrieval (FAISS)"):
            qend = min(len(qm), qstart + qbs)
            q_chunk = np.ascontiguousarray(query[qstart:qend])
            t_chunk_start = time.perf_counter()
            sims_chunk, idx_chunk = index.search(q_chunk, k_eff)
            t_chunk_ms = (time.perf_counter() - t_chunk_start) * 1000.0
            print(
                f"[faiss] chunk {qstart}:{qend} size={qend-qstart} "
                f"search_ms={t_chunk_ms:.2f} qps={(qend-qstart)/(t_chunk_ms/1000.0 + 1e-9):.1f}"
            )
            all_idx[qstart:qend] = idx_chunk
            all_sims[qstart:qend] = sims_chunk
        _neighbors_cache_save(
            npz_path,
            meta_path,
            idx=all_idx,
            sims=all_sims,
            k_eff=k_eff,
            n_bank=len(bank),
            dim=int(bank.shape[1]),
            path_info={
                "bank_embeddings": str(bank_p.resolve()),
                "query_embeddings": str(qemb_p.resolve()),
            },
        )

    results: list[dict[str, Any]] = []
    for qi in tqdm(range(len(qm)), desc="Voting + results"):
        r = qm.iloc[qi]
        uq = str(r["utt_id"])
        expected = str(r["expected_phoneme"])
        top_idx = all_idx[qi]
        top_sim = all_sims[qi]
        nbr_labels = [bank_lab[int(j)] for j in top_idx]
        if vote_pairs or vote_default != 1.0:
            maj_label = weighted_majority_label(nbr_labels, expected, vote_pairs, vote_default)
        else:
            maj_label, _maj_c = Counter(nbr_labels).most_common(1)[0]
        n_exp = sum(1 for l in nbr_labels if l == expected)
        pred_ok = bool(maj_label == expected)
        diagnosis = "" if pred_ok else maj_label
        results.append(
            {
                "query_row": qi,
                "utt_id": uq,
                "phoneme_index": int(r["phoneme_index"]) if "phoneme_index" in r and pd.notna(r["phoneme_index"]) else -1,
                "expected_phoneme": expected,
                "n_neighbors": int(len(top_idx)),
                "neighbor_indices": " ".join(str(int(x)) for x in top_idx),
                "neighbor_phonemes": " ".join(nbr_labels),
                "neighbor_sims": " ".join(f"{float(x):.4f}" for x in top_sim),
                "majority_label": maj_label,
                "n_expected_in_topk": int(n_exp),
                "pred_is_correct": pred_ok,
                "diagnosed_phoneme": diagnosis,
            }
        )

    res_df = pd.DataFrame(results)
    t_retrieval_total_s = time.perf_counter() - t_retrieval_start
    if "gold_is_phoneme_correct" in qm.columns:
        res_df["gold_is_phoneme_correct"] = qm["gold_is_phoneme_correct"].values
        if "gold_realized_phoneme" in qm.columns:
            res_df["gold_realized_phoneme"] = qm["gold_realized_phoneme"].values
        res_df["pred_disagrees_with_gold"] = res_df["pred_is_correct"] != res_df["gold_is_phoneme_correct"]
    elif "is_correct" in qm.columns:
        gold = qm["is_correct"].map(lambda x: str(x).lower() in ("true", "1", "yes"))
        res_df["is_correct"] = qm["is_correct"].values
        res_df["gold_is_phoneme_correct"] = gold.values
        res_df["pred_disagrees_with_gold"] = res_df["pred_is_correct"] != res_df["gold_is_phoneme_correct"]

    res_df.to_csv(out_csv, index=False)

    stats: dict[str, Any] = {
        "script": "05_retrieve_mdd.py",
        "bank_embeddings": str(bank_p),
        "bank_meta": str(bank_meta_p),
        "query_embeddings": str(qemb_p),
        "query_meta": str(qmeta_p),
        "bank_input_mode": bank_load_info["mode"],
        "query_input_mode": query_load_info["mode"],
        "bank_input_files": bank_load_info["files"],
        "query_input_files": query_load_info["files"],
        "vote_weights_path": str(as_abs(args.vote_weights)) if args.vote_weights else None,
        "n_bank": int(len(bank)),
        "n_query": int(len(res_df)),
        "top_k": k,
        "top_k_effective": int(k_eff),
        "retrieval_backend": "faiss_index_flat_ip",
        "faiss_requested_device": str(args.faiss_device),
        "faiss_used_device": faiss_device_used,
        "faiss_num_gpus": int(faiss_num_gpus),
        "faiss_index_type": type(index).__name__,
        "faiss_gpu_id": int(args.faiss_gpu_id),
        "faiss_query_batch_size": int(qbs),
        "neighbors_cache": {
            "npz": str(npz_path),
            "meta": str(meta_path),
            "loaded_from_disk": bool(loaded_cache),
        },
        "retrieval_time_sec": float(t_retrieval_total_s),
        "retrieval_qps": float(len(res_df) / (t_retrieval_total_s + 1e-9)),
        "n_pred_correct": int(res_df["pred_is_correct"].sum()) if len(res_df) else 0,
        "results_csv": str(out_csv),
        "input_definitions": {
            "pair_weight_key": f"expected{PAIR_SEP}neighbor_label",
            "what_is_said": "Annotated/realized phoneme sequence from query metadata (phonemes / derived per-phone realized token).",
            "what_is_predicted": "Model retrieval decision per phoneme (pred_is_correct, diagnosed_phoneme).",
            "what_should_have_been_said": "Reference target phoneme sequence (expected_phoneme / phoneme_ref).",
        },
        "retrieval_crosstab": {
            "description": (
                "Sparse (expected_phoneme, majority_label) counts from k-NN vote. "
                "Use fr_rows when calibrating false-alarm pairs (down-weight majority_label y for expected x)."
            ),
            "expected_vs_majority_all_query_rows": crosstab_expected_vs_majority(res_df),
        },
    }

    if "gold_is_phoneme_correct" in res_df.columns:
        gold_ok = res_df["gold_is_phoneme_correct"].astype(bool)
        pred_ok = res_df["pred_is_correct"].astype(bool)
        ta = int((gold_ok & pred_ok).sum())
        tr = int((~gold_ok & ~pred_ok).sum())
        fr = int((gold_ok & ~pred_ok).sum())
        fa = int((~gold_ok & pred_ok).sum())

        precision = tr / (tr + fr) if (tr + fr) else 0.0
        recall = tr / (tr + fa) if (tr + fa) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        frr = fr / (ta + fr) if (ta + fr) else 0.0
        far = fa / (fa + tr) if (fa + tr) else 0.0

        metrics = {
            "confusion_components": {
                "TA": ta,
                "TR": tr,
                "FR": fr,
                "FA": fa,
            },
            "detection_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            },
            "diagnostic_rates": {
                "FRR": frr,
                "FAR": far,
            },
        }
        stats["phoneme_level_metrics"] = metrics

        fr_only = res_df[gold_ok & ~pred_ok]
        stats["retrieval_crosstab"]["fr_rows_gold_ok_pred_wrong"] = crosstab_expected_vs_majority(fr_only)

        acc = float((pred_ok == gold_ok).mean())
        stats["agreement_with_gold"] = {
            "accuracy_pred_vs_gold": acc,
            "n_rows": int(len(res_df)),
        }
        if "gold_realized_phoneme" in res_df.columns:
            wrong = res_df[gold_ok == False].copy()  # noqa: E712
            if len(wrong):
                diag_ok = wrong["diagnosed_phoneme"].astype(str) == wrong["gold_realized_phoneme"].astype(str)
                cd = int(diag_ok.sum())
                de = int(len(wrong) - cd)
                der = de / (cd + de) if (cd + de) else 0.0
                stats["diagnosis_on_wrong_only"] = {
                    "n_wrong_gold": int(len(wrong)),
                    "CD": cd,
                    "DE": de,
                    "DER": der,
                    "diag_match_rate": float(diag_ok.mean()),
                }
                stats["phoneme_level_metrics"]["diagnostic_rates"]["DER"] = der

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"[timing] total_elapsed_sec={time.perf_counter() - t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
