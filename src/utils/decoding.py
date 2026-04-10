"""CTC decoding utilities."""

from typing import Dict, List

import torch

from src.data.vocab import id2phone


def greedy_decode_ids(logits: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Greedy CTC decode.

    Args:
        logits: (B, T, V) unnormalized logits.
        blank_id: CTC blank token id.
    """
    pred = torch.argmax(logits, dim=-1)  # (B, T)
    results: List[List[int]] = []
    for row in pred:
        seq = row.detach().cpu().tolist()
        collapsed: List[int] = []
        prev = None
        for token in seq:
            if token != prev:
                collapsed.append(token)
            prev = token
        collapsed = [t for t in collapsed if t != blank_id]
        results.append(collapsed)
    return results


def decode_ids_to_phones(
    sequences: List[List[int]],
    vocab: Dict[str, int],
) -> List[List[str]]:
    inv = id2phone(vocab)
    out: List[List[str]] = []
    for seq in sequences:
        toks = []
        for idx in seq:
            tok = inv.get(idx, "<unk>")
            if tok in {"<pad>", "<unk>", "<blank>"}:
                continue
            toks.append(tok)
        out.append(toks)
    return out

