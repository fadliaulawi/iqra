# Plans Light — Step 2: Zero-Training MDD via Confidence / Posterior Analysis

## Objective
Produce a **training-free** MDD system by exploiting the posterior distribution of a **pretrained phoneme recognizer** under forced alignment to the canonical reference. No weights are updated.

## Motivation (paper positioning)
The shared-task paper [`papers/iqraeval.txt`](../papers/iqraeval.txt) describes baselines that all **train** a CTC head on the released phoneme set. We hypothesize that a **non-trained posterior-confidence signal** can already recover a large fraction of the detection capability, which isolates how much of the problem is solvable by representation alone versus by task-specific fitting.

## Prerequisites
- `plans_light/step1.md` done.
- A pretrained **phoneme-level** CTC model is available. Options (no fine-tuning in this step):
  1. `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (multilingual phonemes, IPA-like).
  2. Any Step 2 frozen baseline checkpoint *from the main track* you already trained — used here **as a frozen feature extractor only**, no new training.

## 2.1 — Pipeline
For each test utterance `x` with canonical reference `R = r_1 r_2 ... r_N`:

1. Run the pretrained CTC model to obtain frame log-probs `P ∈ R^{T × V}`.
2. Map the model's phoneme vocabulary to the IqraEval canonical set:
   - Build a fixed, hand-free mapping table once (IPA → IqraEval) and reuse.
   - Phones with no mapping are marked `<skip>` and ignored at scoring time (log this).
3. Perform **forced CTC alignment** to `R` using standard dynamic programming on `P` (no beam search, no training).
4. For each canonical position `i`, collect the aligned time span `[t_s, t_e]` and compute frame-level scores:
   - `post_ri = mean_t P[t, r_i]` (log-prob of the canonical phone).
   - `top1_t  = max_v P[t, v]`
   - `margin_t = top1_t - second_t` (confidence margin).
   - `entropy_t = -Σ_v p_v log p_v`.
5. Reduce per-position to a single **position confidence** `c_i`:
   - `c_i = mean(post_ri) − α * mean(entropy_t) + β * mean(margin_t)`, with α, β ∈ {0, 0.5, 1} (grid, no training).
6. Decision rule: mark position `i` as **mispronounced** iff `c_i < τ`.
   - `τ` chosen on the **dev** set (simple percentile threshold, no gradient-based training).

## 2.2 — Outputs
For each test utterance, emit a predicted phoneme sequence compatible with the evaluation format:
- If position `i` is accepted: output `r_i`.
- If rejected: output the **argmax phone at the center frame of that span** (diagnostic slot).

Save to `outputs/light/step2/predictions.csv` with the standard `ID,Prediction` schema so `scripts/evaluate_predictions.py` works unchanged.

## 2.3 — Ablations
Small and cheap; none require training.
| Ablation | Setting |
|----------|---------|
| A1 | α=β=0 (pure posterior) |
| A2 | α=1, β=0 (entropy only) |
| A3 | α=0, β=1 (margin only) |
| A4 | best (α, β) on dev |
| A5 | remove IPA→IqraEval mapping, keep native model vocab (sanity) |

## Success Criteria
- [ ] End-to-end pipeline runs without any `requires_grad=True` parameters.
- [ ] F1 on released GT is **reported and placed relative to paper baselines** (expected: below baseline 1 but non-trivially above random).
- [ ] Clear per-ablation table; winner identified on dev, not on test.

## Risk / Limits (document honestly)
- IPA→IqraEval mapping introduces an uncontrolled constant; we quantify it via A5.
- Forced alignment quality depends on the pretrained model’s recall on Arabic.

## Output Files
```
src/light/ctc_confidence.py
scripts/light_predict_step2.py
outputs/light/step2/predictions.csv
outputs/light/step2/metrics.txt
outputs/light/step2/ablation.csv
```
