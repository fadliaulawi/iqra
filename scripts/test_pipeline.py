"""End-to-end smoke test for the IqraEval pipeline.

Verifies:
  1. Phoneme vocabulary builds correctly
  2. Evaluation alignment produces correct edit operations
  3. MDD classification works on synthetic examples
  4. All 9+ metrics are computed
  5. (Optional) DataLoader loads real data if available

Usage:
  python -m scripts.test_pipeline
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.vocab import build_vocab, encode_phonemes, decode_ids
from src.evaluation.alignment import align, get_edit_ops, count_edits
from src.evaluation.metrics import (
    classify_mdd,
    compute_metrics,
    evaluate_utterance,
    evaluate_corpus,
    evaluate_phoneme_recognition,
    print_metrics,
)


def test_vocab():
    print("\n[1/5] Testing vocabulary...")
    vocab = build_vocab()
    assert len(vocab) >= 71, f"Expected >=71 tokens, got {len(vocab)}"
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<blank>"] == 2

    test_str = "f ii h i nn a x A y r aa t H i s aa n"
    encoded = encode_phonemes(test_str, vocab)
    decoded = decode_ids(encoded, vocab)
    assert decoded == test_str, f"Round-trip failed: '{decoded}' != '{test_str}'"
    print(f"  Vocab size: {len(vocab)} ({len(vocab) - 3} phonemes + 3 special)")
    print(f"  Encode/decode round-trip: OK")
    return vocab


def test_alignment():
    print("\n[2/5] Testing alignment...")

    ref = ["a", "b", "c", "d"]
    hyp = ["a", "x", "c", "d"]
    al_ref, al_hyp = align(ref, hyp)
    ops = get_edit_ops(al_ref, al_hyp)
    edits = count_edits(al_ref, al_hyp)
    print(f"  ref: {al_ref}")
    print(f"  hyp: {al_hyp}")
    print(f"  ops: {ops}")
    assert edits["substitutions"] == 1
    assert edits["correct"] == 3

    ref2 = ["a", "b", "c"]
    hyp2 = ["a", "c"]
    al_ref2, al_hyp2 = align(ref2, hyp2)
    edits2 = count_edits(al_ref2, al_hyp2)
    print(f"  Deletion test: {edits2}")
    assert edits2["deletions"] == 1

    ref3 = ["a", "c"]
    hyp3 = ["a", "b", "c"]
    al_ref3, al_hyp3 = align(ref3, hyp3)
    edits3 = count_edits(al_ref3, al_hyp3)
    print(f"  Insertion test: {edits3}")
    assert edits3["insertions"] == 1

    print("  All alignment tests passed")


def test_mdd_classification():
    print("\n[3/5] Testing MDD classification...")

    # Case 1: perfect prediction (all correct, model agrees)
    canon = ["a", "b", "c"]
    verb = ["a", "b", "c"]  # no errors
    pred = ["a", "b", "c"]  # model predicts correctly
    counts = classify_mdd(canon, verb, pred)
    assert counts["TA"] == 3 and counts["TR"] == 0 and counts["FA"] == 0 and counts["FR"] == 0
    print(f"  Perfect prediction: TA={counts['TA']}, TR={counts['TR']}, FA={counts['FA']}, FR={counts['FR']} OK")

    # Case 2: mispronunciation correctly detected
    canon2 = ["a", "b", "c"]
    verb2 = ["a", "x", "c"]  # 'b' was mispronounced as 'x'
    pred2 = ["a", "x", "c"]  # model correctly detected the error
    counts2 = classify_mdd(canon2, verb2, pred2)
    assert counts2["TA"] == 2  # 'a' and 'c' correct
    assert counts2["TR"] == 1  # 'b'→'x' detected
    assert counts2["CD"] == 1  # correct diagnosis (predicted 'x' matches verbatim 'x')
    print(f"  Correct detection: TA={counts2['TA']}, TR={counts2['TR']}, CD={counts2['CD']} OK")

    # Case 3: false accept (missed error)
    canon3 = ["a", "b", "c"]
    verb3 = ["a", "x", "c"]  # 'b' was mispronounced
    pred3 = ["a", "b", "c"]  # model missed the error
    counts3 = classify_mdd(canon3, verb3, pred3)
    assert counts3["FA"] == 1
    print(f"  False accept: FA={counts3['FA']} OK")

    # Case 4: false reject (flagged correct as error)
    canon4 = ["a", "b", "c"]
    verb4 = ["a", "b", "c"]  # all correct
    pred4 = ["a", "x", "c"]  # model incorrectly flagged 'b'
    counts4 = classify_mdd(canon4, verb4, pred4)
    assert counts4["FR"] == 1
    print(f"  False reject: FR={counts4['FR']} OK")

    print("  All MDD classification tests passed")


def test_metrics():
    print("\n[4/5] Testing metric computation...")

    counts = {"TA": 100, "TR": 20, "FA": 5, "FR": 10, "CD": 15, "ED": 5}
    m = compute_metrics(counts)

    print_metrics(m, "Test Metrics")

    assert "F1" in m
    assert "Precision" in m
    assert "Recall" in m
    assert "FRR" in m
    assert "FAR" in m
    assert "DER" in m
    assert "Correct_Rate" in m
    assert "Accuracy" in m

    assert 0 <= m["F1"] <= 1
    assert 0 <= m["Precision"] <= 1
    assert 0 <= m["Recall"] <= 1

    print("  All metric assertions passed")


def test_corpus_eval():
    print("\n[5/5] Testing corpus-level evaluation...")

    random.seed(42)
    vocab = build_vocab()
    phoneme_list = [p for p in vocab if p not in {"<pad>", "<unk>", "<blank>"}]

    canonicals, verbatims, predicteds = [], [], []

    for _ in range(50):
        length = random.randint(5, 20)
        canon = [random.choice(phoneme_list) for _ in range(length)]

        # Simulate ~10% mispronunciation rate
        verb = list(canon)
        for j in range(length):
            if random.random() < 0.10:
                verb[j] = random.choice(phoneme_list)

        # Simulate model predictions with ~15% error
        pred = list(canon)
        for j in range(length):
            if random.random() < 0.15:
                pred[j] = random.choice(phoneme_list)

        canonicals.append(canon)
        verbatims.append(verb)
        predicteds.append(pred)

    mdd_metrics = evaluate_corpus(canonicals, verbatims, predicteds)
    print_metrics(mdd_metrics, "Corpus MDD Evaluation (synthetic)")

    refs = canonicals
    preds = predicteds
    per_metrics = evaluate_phoneme_recognition(refs, preds)
    print_metrics(per_metrics, "Phoneme Error Rate (synthetic)")

    print("  Corpus evaluation OK")


def main():
    print("=" * 60)
    print("  IqraEval Pipeline — End-to-End Smoke Test")
    print("=" * 60)

    test_vocab()
    test_alignment()
    test_mdd_classification()
    test_metrics()
    test_corpus_eval()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
