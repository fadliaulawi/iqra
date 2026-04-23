# Plans Light — Mini Report: Positioning vs the IqraEval Paper

This report frames the **plans-light track** as a candidate publication that complements, rather than competes on raw F1 with, the IqraEval shared-task systems.

## 1. Setting
- Source paper: IqraEval 2025 shared task (see [`papers/iqraeval.txt`](../papers/iqraeval.txt)).
- Shared-task baseline 1 (mHuBERT) reports **F1 = 0.4414** on the open test set.
- Top submitted system (BAIC) reports **F1 = 0.4726** with extra synthetic data and task-adaptive pretraining.

All of those systems share a common recipe:
**SSL/ASR backbone → supervised phoneme head → CTC fine-tuning → alignment-based MDD evaluation.**

Our main-track pipeline (`plans/`) is a direct variant of that recipe; we use it as our **internal control**.

The plans-light track proposes a different axis of comparison: **how much of IqraEval can be solved with no (or near-no) training?**

## 2. Core Argument
We argue three claims, each independently publishable:

1. **The problem is largely a representation-geometry problem.**  
   A frozen SSL encoder + a per-phoneme anchor bank already separates correct from mispronounced segments with non-trivial F1 on IqraEval (`plans_light/step3.md`).

2. **Pretrained posteriors carry rich detection signal.**  
   Forced-alignment confidence, margin, and entropy from a generic phoneme model reach useful F1 without any gradient update on IqraEval data (`plans_light/step2.md`).

3. **The remaining gap to trained baselines is mostly calibration, not representation.**  
   A small supervised head (≤1M params, frozen encoder) closes a substantial part of the gap to the trained baseline 1 (`plans_light/step4.md`), suggesting the backbone already contains the needed information.

## 3. Claimed Contributions
- **C1 — Anchor-Bank MDD:** a simple, reproducible per-phoneme reference-embedding method that treats MDD as Mahalanobis/cosine deviation against a canonical anchor, with multi-layer (acoustic+linguistic) fusion.
- **C2 — Zero-Training MDD Benchmark:** a well-defined *training-free* baseline on the official IqraEval open test set, something the shared-task paper does not provide.
- **C3 — Gap Decomposition:** a quantitative analysis of how much of the F1 gap between zero-training methods and the organizers’ baseline 1 is closable by a 3-layer MLP on diff-features, with the encoder frozen.
- **C4 — Ablation Axes orthogonal to the shared task:** mapping quality, alignment choice, layer depth, encoder family, and anchor-bank size — none of which are studied in the shared-task paper.
- **C5 — Reference-Quality Ablation:** a clean isolation of how much of our zero-training F1 comes from the *reference transcription* vs. the *representation*, enabled by swapping in the Tajweed-aware phonemizer (Ibrahim et al., OpenReview `hZt0JK28iV`) with a versioned 71→62 crosswalk. See `plans_light/step5.md`.

## 4. Why Reviewers Should Care Even Without SOTA F1
- **Reproducibility at low cost.** Steps 1–3 require no fine-tuning and run on modest hardware.
- **Analysis value.** The framework is a probe into what pretrained speech encoders already know about Arabic pronunciation, which generalizes beyond IqraEval.
- **Method novelty.** Neither the IqraEval 2025 paper nor the submitted teams study the training-free regime explicitly; our framing converts an engineering race into a scientific question.
- **Cross-task transfer.** The Anchor-Bank + confidence recipe is directly applicable to other phoneme-level MDD corpora (L2-Arctic, CU-CHLOE, etc.), making our contribution reusable outside Qur’anic recitation.

## 5. Expected Empirical Picture (intentionally modest)
We do **not** claim to beat the top leaderboard entries. Our target empirical story is:

- Step 2 (posterior-only, training-free): F1 clearly above naive/random; below baseline 1.
- Step 3 (anchor distance, training-free): comparable or better than Step 2, with clearer ablation insight.
- Step 2+3 fusion (still training-free): best zero-training number, the headline of the paper.
- Step 4 (tiny head): reduces the gap to baseline 1 substantially with ≤1M trainable parameters and ≤1 GPU-hour, used **as a bounding experiment**, not as the main claim.

Our main-track `plans/` runs are reported alongside as an **internal ceiling**, honestly labeled as “matched recipe to the shared-task baseline.” We do not sell those numbers as this work’s contribution.

## 6. Reviewer Objections and Our Replies
| Objection | Reply |
|-----------|-------|
| “Your F1 is below baseline 1.” | Correct; our baseline is not F1 but the *training-free* regime, which baseline 1 does not occupy. |
| “IPA-to-IqraEval mapping is lossy.” | Quantified via an ablation (A5 in Step 2) so the cost is explicit. |
| “Anchor bank = memorization.” | We show it is (i) tiny in memory, (ii) built only from correct training data, (iii) robust to encoder swap — none of which are properties of a classifier that memorizes labels. |
| “What’s new over zero-shot ASR-confidence?” | We add multi-layer anchor distance (representation geometry) and quantify its complementarity to posterior confidence through the fusion ablation. |

## 7. Ethics / Data
- All data from the IqraEval Hugging Face organization, used under their terms.
- No additional human annotation.
- Models used frozen; any checkpoint we save is small (<1M params) and clearly marked as research-only.

## 8. Next Steps After This Plan
1. Implement `plans_light/step1.md` → `plans_light/step4.md` in order.
2. Produce `outputs/light/final_table.csv` with all ablations.
3. Draft paper: introduction → related work (centered on IqraEval paper) → method (Sections 2, 3, 4 of this track) → analysis (fusion + gap decomposition) → limits.
4. Submit to an Arabic-NLP or speech-representation venue (e.g., ArabicNLP workshop, Interspeech short paper, SLT).

