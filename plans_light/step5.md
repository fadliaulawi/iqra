# Plans Light — Step 5: Tajweed-Aware Reference Augmentation (Zero-Training)

## Objective
Swap the canonical reference transcription with a **Tajweed-aware phoneme sequence** produced by the open-source Qur’anic phonemizer (Ibrahim, Shahin, Ahmed; OpenReview `hZt0JK28iV`, code [`Hetchy/Quranic-Phonemizer`](https://github.com/Hetchy/Quranic-Phonemizer)), and re-run the zero-training MDD stack.  
This step keeps the **light-research contract**: no model training, no IqraEval-specific weights updated.

## Motivation
- The IqraEval shared-task pipeline uses the Halabi–Wald phonemizer (62 phones) which does **not** explicitly encode Tajweed phenomena such as Idgham, Iqlab, Ikhfaa, Qalqala, Tafkheem, Waqf.
- The 71-symbol Tajweed-aware inventory is published as a **standalone Python API**. Integrating it is purely on the reference side — ideal for zero-training research.
- This isolates a question the IqraEval paper does not answer: **how much of MDD is a representation problem vs. a reference-quality problem?**

## Prerequisites
- `plans_light/step1.md` (anchor bank) done.
- `plans_light/step2.md` and `plans_light/step3.md` runnable with the default 62-phoneme reference.
- Raw **Qur’anic text** for each training/test utterance (needed for G2P). If only `Reference_phn` is stored, also persist the source `text`/`sentence` column when re-running data download.

## 5.1 — Integrating the Phonemizer
- Install: `pip install git+https://github.com/Hetchy/Quranic-Phonemizer.git` (or vendor into `third_party/quranic_phonemizer/`).
- Add `src/light/phonemizer_tajweed.py`:
  - `text_to_phones_tajweed(text: str) -> List[str]`
  - `tajweed_to_iqra62(phones: List[str]) -> List[str]` — **deterministic crosswalk** from the 71-symbol Tajweed set to the IqraEval 62-set; document every collapse rule and unknown symbol.
- Cache phonemized sequences on disk to keep inference fast and deterministic:
  - `data/processed/tajweed_phones/{train,dev,test}.parquet` with columns `ID, text, phones_tajweed, phones_iqra62`.

## 5.2 — Variants to Build
| Variant | Canonical reference | Anchor bank keys | Decision space |
|---------|---------------------|------------------|----------------|
| V0 (baseline) | IqraEval `Reference_phn` (62) | 62-set | 62-set |
| V1 | Tajweed phones (71), collapsed → 62 for eval | 62-set | 62-set |
| V2 | Tajweed phones (71), native | **71-set** | 71-set; collapse predictions to 62 at scoring |
| V3 | Tajweed phones (71), native | 71-set | 71-set; dual reporting (native + collapsed-62) |

## 5.3 — Scoring Contract
- **Leaderboard-comparable numbers** must always be reported in the **IqraEval 62-set** via the crosswalk, using `scripts/evaluate_predictions.py` unchanged.
- **Native 71-set numbers** are reported as a *diagnostic*, not as a competitive score.

## 5.4 — Ablations (all training-free)
| Ablation | What changes |
|----------|--------------|
| T1 | V0 vs V1 on Step 2 (posterior confidence) |
| T2 | V0 vs V2 on Step 3 (anchor distance) |
| T3 | V0 vs V3 on Step 2+3 fusion |
| T4 | Sensitivity to the crosswalk: randomly perturb 5% of mappings and re-score |
| T5 | Coverage audit: per-utterance rate of Tajweed phones that have no 62-set mapping |

## 5.5 — Light-Training Touchpoint (optional)
Only if Step 4 has already been run:
- Re-train the same tiny MLP head from `plans_light/step4.md` on **V1 features** (Tajweed-reference-derived diff features, collapsed to 62 for labels).
- This strictly isolates whether the Tajweed reference helps the **calibration layer**, holding everything else fixed.

## Success Criteria
- [ ] V0–V3 each produce a complete predictions CSV and a metrics row.
- [ ] V1/V2/V3 improve at least one of F1 / FAR / FRR over V0 on the dev set before any test-set number is reported.
- [ ] Crosswalk is versioned (`tajweed_to_iqra62.json`) and coverage audit printed.
- [ ] No IqraEval-trained weights were updated in this step.

## Risks / Limits (document honestly)
- Tajweed G2P presumes **Hafs ʿan ʿAsim**; utterances outside that style may be mis-phonemized.
- The 71→62 crosswalk is deterministic and necessarily loses information; V2/V3 quantify that loss.
- This step does **not** claim any novel phonological work; the contribution is purely empirical integration and analysis.

## Output Files
```
src/light/phonemizer_tajweed.py
scripts/light_build_tajweed_phones.py
scripts/light_predict_step5_v1.py
scripts/light_predict_step5_v2.py
scripts/light_predict_step5_v3.py
outputs/light/step5/V0_metrics.txt
outputs/light/step5/V1_metrics.txt
outputs/light/step5/V2_metrics.txt
outputs/light/step5/V3_metrics.txt
outputs/light/step5/ablation.csv
outputs/light/tajweed_to_iqra62.json
```
