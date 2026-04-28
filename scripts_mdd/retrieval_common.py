"""Shared helpers for MDD retrieval scripts (04 train, 05 inference)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def as_abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_embeddings_with_parts(
    emb_path: Path, meta_path: Path, role: str
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load single embedding/meta pair or auto-merge *.partXofY files."""
    info: dict[str, Any] = {"mode": "single", "files": []}
    if emb_path.exists() and meta_path.exists():
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        info["files"] = [{"embeddings": str(emb_path), "meta": str(meta_path)}]
        return emb, meta, info

    emb_parts = sorted(emb_path.parent.glob(f"{emb_path.stem}.part*of*.npy"))
    meta_parts = sorted(meta_path.parent.glob(f"{meta_path.stem}.part*of*.csv"))
    if not emb_parts or not meta_parts:
        raise SystemExit(
            f"Missing {role} files. Checked single ({emb_path}, {meta_path}) and part patterns in parent dirs."
        )

    def _suffix_no_ext(p: Path) -> str:
        s = p.stem
        i = s.find(".part")
        return s[i:] if i >= 0 else ""

    meta_by_suffix = {_suffix_no_ext(p): p for p in meta_parts}
    emb_list: list[np.ndarray] = []
    meta_list: list[pd.DataFrame] = []
    for ep in emb_parts:
        suf = _suffix_no_ext(ep)
        mp = meta_by_suffix.get(suf)
        if mp is None:
            raise SystemExit(f"Found embedding part {ep} but no matching meta part with suffix {suf}")
        emb_list.append(np.load(ep))
        meta_list.append(pd.read_csv(mp))
        info["files"].append({"embeddings": str(ep), "meta": str(mp)})
    info["mode"] = "parts"
    emb = np.concatenate(emb_list, axis=0)
    meta = pd.concat(meta_list, axis=0, ignore_index=True)
    return emb, meta, info


def align_ref_hyp(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Needleman-Wunsch style alignment (unit costs) returning aligned token pairs."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, n + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            if best == sub:
                bt[i][j] = 0
            elif best == dele:
                bt[i][j] = 1
            else:
                bt[i][j] = 2
    out: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            out.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bt[i][j] == 1):
            out.append((ref[i - 1], None))
            i -= 1
        else:
            out.append((None, hyp[j - 1]))
            j -= 1
    out.reverse()
    return out


def derive_phoneme_gold(qm: pd.DataFrame) -> pd.DataFrame:
    """Add gold_is_phoneme_correct/gold_realized_phoneme from phoneme_ref vs phonemes if available."""
    need_cols = {"utt_id", "phoneme_index", "phoneme_ref", "phonemes"}
    if not need_cols.issubset(set(qm.columns)):
        return qm
    if "gold_is_phoneme_correct" in qm.columns:
        return qm

    qm = qm.copy()
    qm["gold_is_phoneme_correct"] = pd.Series(pd.array([pd.NA] * len(qm), dtype="boolean"), index=qm.index)
    qm["gold_realized_phoneme"] = ""

    for utt_id, g in qm.groupby("utt_id", sort=False):
        g_idx = g.index.tolist()
        ref_s = ""
        hyp_s = ""
        for ridx in g_idx:
            rref = str(qm.at[ridx, "phoneme_ref"]) if pd.notna(qm.at[ridx, "phoneme_ref"]) else ""
            rhyp = str(qm.at[ridx, "phonemes"]) if pd.notna(qm.at[ridx, "phonemes"]) else ""
            if rref and rref.lower() != "nan":
                ref_s = rref
            if rhyp and rhyp.lower() != "nan":
                hyp_s = rhyp
            if ref_s and hyp_s:
                break
        ref = [t for t in ref_s.split() if t]
        hyp = [t for t in hyp_s.split() if t]
        if not ref:
            continue
        aln = align_ref_hyp(ref, hyp)
        ref_map: list[tuple[bool, str]] = []
        for a_ref, a_hyp in aln:
            if a_ref is None:
                continue
            if a_hyp is None:
                ref_map.append((False, "<del>"))
            else:
                ref_map.append((a_ref == a_hyp, a_hyp))
        for ridx in g_idx:
            try:
                pidx = int(qm.at[ridx, "phoneme_index"])
            except Exception:
                continue
            if 0 <= pidx < len(ref_map):
                ok, realized = ref_map[pidx]
                qm.at[ridx, "gold_is_phoneme_correct"] = bool(ok)
                qm.at[ridx, "gold_realized_phoneme"] = realized
    if qm["gold_is_phoneme_correct"].notna().any():
        qm["gold_is_phoneme_correct"] = qm["gold_is_phoneme_correct"].fillna(False).astype(bool)
    return qm


def crosstab_expected_vs_majority(res_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Sparse counts: expected_phoneme x k-NN majority label."""
    if res_df.empty or "expected_phoneme" not in res_df.columns or "majority_label" not in res_df.columns:
        return []
    g = res_df.groupby(["expected_phoneme", "majority_label"], dropna=False, sort=False).size()
    out = [
        {"expected_phoneme": str(e), "majority_label": str(m), "count": int(c)}
        for (e, m), c in g.items()
    ]
    out.sort(key=lambda d: -d["count"])
    return out


PAIR_SEP = "||"


def pair_key(expected: str, neighbor_label: str) -> str:
    return f"{expected}{PAIR_SEP}{neighbor_label}"


def load_vote_weights_json(path: Path) -> tuple[dict[str, float], float]:
    """Load { default_weight, pairs: { "x||y": w, ... } }."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    default = float(data.get("default_weight", 1.0))
    raw = data.get("pairs", {}) or {}
    pairs = {str(k): float(v) for k, v in raw.items()}
    return pairs, default


def weight_for_pair(
    pairs: dict[str, float], default_w: float, expected: str, neighbor_label: str
) -> float:
    return float(pairs.get(pair_key(expected, neighbor_label), default_w))


def weighted_majority_label(nbr_labels: list[str], expected: str, pairs: dict[str, float], default_w: float) -> str:
    """Each neighbor vote for its label, weighted by w(expected, label). Tie-break: earlier in neighbor list."""
    scores: dict[str, float] = {}
    first_i: dict[str, int] = {}
    for i, lab in enumerate(nbr_labels):
        w = weight_for_pair(pairs, default_w, expected, lab)
        scores[lab] = scores.get(lab, 0.0) + w
        if lab not in first_i:
            first_i[lab] = i
    return max(scores.keys(), key=lambda lab: (scores[lab], -first_i[lab]))


def build_faiss_index(
    bank: np.ndarray, faiss_device: str, faiss_gpu_id: int
) -> tuple[Any, str, int]:
    """Return (faiss index, used_device, n_gpus)."""
    import faiss  # type: ignore

    faiss_num_gpus = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
    cpu_index = faiss.IndexFlatIP(int(bank.shape[1]))
    faiss_device_used = "cpu"
    index: Any = cpu_index
    if faiss_device in ("auto", "gpu") and faiss_num_gpus > 0:
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            if faiss_device == "gpu":
                raise SystemExit("Requested --faiss-device gpu, but this faiss build lacks GPU bindings.")
            print("[faiss] GPU detected but GPU bindings not available; falling back to CPU.")
        else:
            gpu_id = int(faiss_gpu_id)
            if gpu_id < 0 or gpu_id >= faiss_num_gpus:
                raise SystemExit(
                    f"--faiss-gpu-id {gpu_id} out of range; available GPUs: 0..{faiss_num_gpus-1}"
                )
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, cpu_index)
            faiss_device_used = f"gpu:{gpu_id}"
    elif faiss_device == "gpu" and (faiss_num_gpus == 0 or not hasattr(faiss, "index_cpu_to_gpu")):
        raise SystemExit("Requested --faiss-device gpu, but FAISS GPU is not available for this build.")
    return index, faiss_device_used, faiss_num_gpus
