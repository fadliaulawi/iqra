"""Full MDD (Mispronunciation Detection & Diagnosis) evaluation.

Implements the official IqraEval evaluation protocol:
  1. Three-way alignment: canonical × verbatim × predicted
  2. Classification into TA / TR / FA / FR / CD / ED
  3. Computation of F1, Precision, Recall, FRR, FAR, DER, Correct Rate, Accuracy

Reference: IqraEval 2025 shared task paper, Section 2.3
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.alignment import align, count_edits, GAP


def classify_mdd(
    canonical: List[str],
    verbatim: List[str],
    predicted: List[str],
) -> Dict[str, int]:
    """Three-way alignment and MDD classification for a single utterance.

    Args:
        canonical: reference (text-dependent) phoneme sequence
        verbatim:  annotated (what was actually said) phoneme sequence
        predicted: model's predicted phoneme sequence

    Returns:
        Counts of TA, TR, FA, FR, CD, ED
    """
    # Step 1: align canonical with verbatim to find where errors are
    canon_al, verb_al = align(canonical, verbatim)

    # Step 2: align canonical with predicted
    canon_al2, pred_al = align(canonical, predicted)

    # Build position-level lookup: for each canonical position, what did
    # verbatim say and what did predicted say?
    # We need to align all three. Strategy: use canonical as anchor.

    # Simpler approach matching the official eval:
    # Align (canonical, verbatim) → identify correct vs error positions
    # Align (canonical, predicted) → identify what model predicted
    # Then cross-reference by canonical position index.

    # Flatten canonical positions from both alignments
    # Position in canonical: track which canonical phone we're at
    canon_verb_map = {}  # canon_pos -> verbatim phone (or None if deletion)
    cpos = 0
    for c, v in zip(canon_al, verb_al):
        if c != GAP:
            canon_verb_map[cpos] = v
            cpos += 1
        # if c == GAP, verbatim inserted extra phone — doesn't map to canon position

    canon_pred_map = {}
    cpos = 0
    for c, p in zip(canon_al2, pred_al):
        if c != GAP:
            canon_pred_map[cpos] = p
            cpos += 1

    ta, tr, fa, fr, cd, ed = 0, 0, 0, 0, 0, 0

    n_canon = len(canonical)
    for pos in range(n_canon):
        verb_phone = canon_verb_map.get(pos, GAP)
        pred_phone = canon_pred_map.get(pos, GAP)
        canon_phone = canonical[pos]

        is_correct = (verb_phone == canon_phone)
        pred_correct = (pred_phone == canon_phone)

        if is_correct and pred_correct:
            ta += 1
        elif is_correct and not pred_correct:
            fr += 1
        elif not is_correct and not pred_correct:
            tr += 1
            if pred_phone == verb_phone:
                cd += 1
            else:
                ed += 1
        elif not is_correct and pred_correct:
            fa += 1

    return {"TA": ta, "TR": tr, "FA": fa, "FR": fr, "CD": cd, "ED": ed}


def compute_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    """Compute all MDD metrics from aggregated counts."""
    ta = counts["TA"]
    tr = counts["TR"]
    fa = counts["FA"]
    fr = counts["FR"]
    cd = counts["CD"]
    ed = counts["ED"]

    # Error rates
    frr = fr / (ta + fr) if (ta + fr) > 0 else 0.0
    far = fa / (fa + tr) if (fa + tr) > 0 else 0.0
    der = ed / (cd + ed) if (cd + ed) > 0 else 0.0

    # Detection metrics
    precision = tr / (tr + fr) if (tr + fr) > 0 else 0.0
    recall = tr / (tr + fa) if (tr + fa) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total = ta + tr + fa + fr
    correct_rate = (ta) / (ta + fr + fa + tr) if total > 0 else 0.0
    accuracy = (ta + tr) / total if total > 0 else 0.0

    return {
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "FRR": round(frr, 4),
        "FAR": round(far, 4),
        "DER": round(der, 4),
        "Correct_Rate": round(correct_rate, 4),
        "Accuracy": round(accuracy, 4),
        "TA": ta,
        "TR": tr,
        "FA": fa,
        "FR": fr,
        "CD": cd,
        "ED": ed,
    }


def evaluate_utterance(
    canonical: List[str],
    verbatim: List[str],
    predicted: List[str],
) -> Dict[str, float]:
    """Evaluate a single utterance."""
    counts = classify_mdd(canonical, verbatim, predicted)
    return compute_metrics(counts)


def evaluate_corpus(
    canonicals: List[List[str]],
    verbatims: List[List[str]],
    predicteds: List[List[str]],
) -> Dict[str, float]:
    """Evaluate over a corpus, aggregating counts before computing metrics."""
    agg = {"TA": 0, "TR": 0, "FA": 0, "FR": 0, "CD": 0, "ED": 0}

    for canon, verb, pred in zip(canonicals, verbatims, predicteds):
        counts = classify_mdd(canon, verb, pred)
        for k in agg:
            agg[k] += counts[k]

    return compute_metrics(agg)


def evaluate_from_csv(
    truth_csv: str,
    pred_csv: str,
    ref_col: str = "Reference_phn",
    ann_col: str = "Annotation_phn",
    pred_col: str = "Prediction",
    id_col: str = "ID",
) -> Dict[str, float]:
    """Evaluate from CSV files matching the IqraEval submission format.

    truth_csv must have columns: ID, Reference_phn, Annotation_phn
    pred_csv must have columns: ID, Prediction
    """
    truth_df = pd.read_csv(truth_csv)
    pred_df = pd.read_csv(pred_csv)

    merged = pd.merge(truth_df, pred_df, on=id_col)

    canonicals = [row[ref_col].split() for _, row in merged.iterrows()]
    verbatims = [row[ann_col].split() for _, row in merged.iterrows()]
    predicteds = [row[pred_col].split() for _, row in merged.iterrows()]

    return evaluate_corpus(canonicals, verbatims, predicteds)


def evaluate_phoneme_recognition(
    references: List[List[str]],
    predictions: List[List[str]],
) -> Dict[str, float]:
    """Simpler evaluation for phoneme recognition (no MDD, just PER/accuracy).

    Useful during training when we don't have verbatim annotations.
    """
    total_ref = 0
    total_errors = 0

    for ref, pred in zip(references, predictions):
        aligned_ref, aligned_pred = align(ref, pred)
        edits = count_edits(aligned_ref, aligned_pred)
        total_errors += edits["substitutions"] + edits["deletions"] + edits["insertions"]
        total_ref += len(ref)

    per = total_errors / total_ref if total_ref > 0 else 1.0
    return {
        "PER": round(per, 4),
        "Accuracy": round(1.0 - per, 4),
        "total_ref_phones": total_ref,
        "total_errors": total_errors,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}  {v:.4f}")
        else:
            print(f"  {k:15s}  {v}")
    print(f"{'='*50}")
