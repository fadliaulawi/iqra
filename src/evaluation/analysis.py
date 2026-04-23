"""Per-phoneme analysis for IqraEval MDD outputs.

Computes success proportion and MDD metrics per canonical phoneme using:
  - truth CSV: ID, Reference_phn, Annotation_phn
  - pred CSV:  ID, Prediction
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.alignment import GAP, align


def _iter_position_events(canonical: List[str], verbatim: List[str], predicted: List[str]):
    """Yield position-level events anchored on canonical phoneme positions."""
    canon_al, verb_al = align(canonical, verbatim)
    canon_al2, pred_al = align(canonical, predicted)

    canon_verb_map = {}
    cpos = 0
    for c, v in zip(canon_al, verb_al):
        if c != GAP:
            canon_verb_map[cpos] = v
            cpos += 1

    canon_pred_map = {}
    cpos = 0
    for c, p in zip(canon_al2, pred_al):
        if c != GAP:
            canon_pred_map[cpos] = p
            cpos += 1

    for pos, canon_phone in enumerate(canonical):
        verb_phone = canon_verb_map.get(pos, GAP)
        pred_phone = canon_pred_map.get(pos, GAP)
        is_correct = verb_phone == canon_phone
        pred_correct = pred_phone == canon_phone

        if is_correct and pred_correct:
            event = "TA"
            diag = None
        elif is_correct and not pred_correct:
            event = "FR"
            diag = None
        elif (not is_correct) and pred_correct:
            event = "FA"
            diag = None
        else:
            event = "TR"
            diag = "CD" if pred_phone == verb_phone else "ED"

        yield canon_phone, event, diag


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def analyze_per_phoneme(truth_csv: str, pred_csv: str, id_col: str = "ID") -> pd.DataFrame:
    truth = pd.read_csv(truth_csv)
    pred = pd.read_csv(pred_csv)

    truth[id_col] = truth[id_col].astype(str)
    pred[id_col] = pred[id_col].astype(str)

    truth_ids = set(truth[id_col].tolist())
    pred_ids = set(pred[id_col].tolist())
    missing = sorted(truth_ids - pred_ids)
    extra = sorted(pred_ids - truth_ids)
    if missing:
        print(f"[WARN] Missing prediction IDs: {len(missing)} (showing first 10) {missing[:10]}")
    if extra:
        print(f"[WARN] Extra prediction IDs: {len(extra)} (showing first 10) {extra[:10]}")

    merged = pd.merge(
        truth[[id_col, "Reference_phn", "Annotation_phn"]],
        pred[[id_col, "Prediction"]],
        on=id_col,
        how="inner",
    )

    agg: Dict[str, Dict[str, int]] = {}
    for _, row in merged.iterrows():
        canonical = str(row["Reference_phn"]).split()
        verbatim = str(row["Annotation_phn"]).split()
        predicted = str(row["Prediction"]).split()

        for phone, event, diag in _iter_position_events(canonical, verbatim, predicted):
            if phone not in agg:
                agg[phone] = {"TA": 0, "TR": 0, "FA": 0, "FR": 0, "CD": 0, "ED": 0}
            agg[phone][event] += 1
            if diag is not None:
                agg[phone][diag] += 1

    rows = []
    for phone, c in sorted(agg.items(), key=lambda kv: kv[0]):
        ta, tr, fa, fr, cd, ed = c["TA"], c["TR"], c["FA"], c["FR"], c["CD"], c["ED"]
        total = ta + tr + fa + fr

        precision = _safe_div(tr, tr + fr)
        recall = _safe_div(tr, tr + fa)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        frr = _safe_div(fr, ta + fr)
        far = _safe_div(fa, fa + tr)
        der = _safe_div(ed, cd + ed)

        # How often this phoneme's canonical positions are classified correctly
        success_prop = _safe_div(ta + tr, total)
        # Correct-pronunciation acceptance quality for this phoneme
        accept_success = _safe_div(ta, ta + fr)
        # Mispronunciation detection quality for this phoneme
        reject_success = _safe_div(tr, tr + fa)

        rows.append(
            {
                "phoneme": phone,
                "support": total,
                "success_proportion": round(success_prop, 4),
                "accept_success": round(accept_success, 4),
                "reject_success": round(reject_success, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "frr": round(frr, 4),
                "far": round(far, 4),
                "der": round(der, 4),
                "TA": ta,
                "TR": tr,
                "FA": fa,
                "FR": fr,
                "CD": cd,
                "ED": ed,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["f1", "success_proportion", "support"], ascending=[False, False, False]).reset_index(
            drop=True
        )
    return df


def main():
    p = argparse.ArgumentParser(description="Per-phoneme MDD analysis.")
    p.add_argument("--truth", default="data/raw/test_gt/ground_truth.csv")
    p.add_argument("--pred", required=True)
    p.add_argument("--output-csv", default=None, help="Defaults to <pred_dir>/analysis_per_phoneme.csv")
    p.add_argument("--top-k", type=int, default=15)
    args = p.parse_args()

    out_csv = args.output_csv
    if out_csv is None:
        out_csv = str(Path(args.pred).resolve().parent / "analysis_per_phoneme.csv")

    df = analyze_per_phoneme(args.truth, args.pred)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved per-phoneme analysis: {out_csv}")
    print(f"Phonemes analyzed: {len(df)}")
    if len(df) > 0:
        k = min(args.top_k, len(df))
        print(f"\nTop {k} phonemes by F1:")
        print(df.head(k)[["phoneme", "support", "f1", "success_proportion", "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()

