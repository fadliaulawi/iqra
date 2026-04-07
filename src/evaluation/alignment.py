"""Needleman-Wunsch global alignment for phoneme sequences.

Compatible with the official IqraEval alignment logic
(s3prl/interspeech_IqraEval/mdd_eval/metric.py) but cleaned up
for direct use in our pipeline.
"""

from typing import List, Tuple

GAP = "<eps>"
GAP_PENALTY = -1
MATCH_AWARD = 1
MISMATCH_PENALTY = -1


def _score(a: str, b: str) -> int:
    if a == b:
        return MATCH_AWARD
    if a == GAP or b == GAP:
        return GAP_PENALTY
    return MISMATCH_PENALTY


def align(seq1: List[str], seq2: List[str]) -> Tuple[List[str], List[str]]:
    """Global alignment of two phoneme sequences via Needleman-Wunsch.

    Args:
        seq1: reference sequence
        seq2: hypothesis sequence

    Returns:
        (aligned_seq1, aligned_seq2) with <eps> gaps inserted.
    """
    n, m = len(seq1), len(seq2)

    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        score[i][0] = GAP_PENALTY * i
    for j in range(n + 1):
        score[0][j] = GAP_PENALTY * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + _score(seq1[j - 1], seq2[i - 1])
            delete = score[i - 1][j] + GAP_PENALTY
            insert = score[i][j - 1] + GAP_PENALTY
            score[i][j] = max(match, delete, insert)

    align1, align2 = [], []
    i, j = m, n

    while i > 0 and j > 0:
        sc = score[i][j]
        if sc == score[i - 1][j - 1] + _score(seq1[j - 1], seq2[i - 1]):
            align1.append(seq1[j - 1])
            align2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif sc == score[i][j - 1] + GAP_PENALTY:
            align1.append(seq1[j - 1])
            align2.append(GAP)
            j -= 1
        else:
            align1.append(GAP)
            align2.append(seq2[i - 1])
            i -= 1

    while j > 0:
        align1.append(seq1[j - 1])
        align2.append(GAP)
        j -= 1
    while i > 0:
        align1.append(GAP)
        align2.append(seq2[i - 1])
        i -= 1

    return align1[::-1], align2[::-1]


def get_edit_ops(aligned1: List[str], aligned2: List[str]) -> List[str]:
    """Classify each aligned position as C(orrect), S(ubstitution), D(eletion), I(nsertion)."""
    ops = []
    for a, b in zip(aligned1, aligned2):
        if a == b:
            ops.append("C")
        elif a != GAP and b == GAP:
            ops.append("D")
        elif a == GAP and b != GAP:
            ops.append("I")
        else:
            ops.append("S")
    return ops


def count_edits(aligned1: List[str], aligned2: List[str]) -> dict:
    ops = get_edit_ops(aligned1, aligned2)
    return {
        "correct": ops.count("C"),
        "substitutions": ops.count("S"),
        "deletions": ops.count("D"),
        "insertions": ops.count("I"),
    }
