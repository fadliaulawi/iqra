# Step 6 — Full Solution: Combine Best Findings + Decoding + Ensemble

## Objective
Combine the best backbone, training strategy, and data mix from steps 2–5. Add decoding improvements (beam search) and optional model ensembling. This is the final submission-ready system. Target: F1 > 0.50.

## Prerequisites
- Steps 2–5 complete
- Ablation results identifying: best backbone, best stage count, best data mix

## 6.1 — Best Single Model

Assemble the best configuration based on ablation results:

```yaml
# configs/step6_best_single.yaml (template — fill from ablation)
encoder_name: "<best from step3 ablation>"
freeze_encoder: false
gradient_checkpointing: true
head_type: "bilstm"
head_hidden: 1024
head_layers: 2

# Multi-stage (if step 4 showed gains)
stages: [stage1, stage2, stage3]

# Data (from step 5)
train_datasets:
  - path: "data/raw/Iqra_train"
    weight: 1.0
  - path: "<best TTS/synth mix from step5>"
    weight: "<best weight>"

# Training (from best settings)
encoder_lr: "<from ablation>"
head_lr: "<from ablation>"
max_epochs: "<from ablation>"
fp16: true
specaugment: true
speed_perturbation: true
```

Retrain from scratch with this combined config to get the cleanest checkpoint.

## 6.2 — Beam Search Decoding

Upgrade from greedy to beam search CTC decoding.

Update `src/utils/decoding.py`:
- Implement prefix beam search for CTC
- Parameters: beam_width = [5, 10, 20]
- Optionally add shallow phoneme language model (trained on canonical phoneme sequences from training data)
  - n-gram LM (3-gram or 5-gram) over phoneme sequences
  - Integrate via shallow fusion: `score = log_ctc + alpha * log_lm`
  - Tune alpha on dev set: try [0.0, 0.1, 0.3, 0.5]

| Run | Decoding | LM | Expected impact |
|-----|----------|-----|----------------|
| greedy | greedy | none | baseline |
| beam5  | beam w=5 | none | +0.005–0.01 |
| beam10 | beam w=10 | none | +0.005–0.015 |
| beam10_lm | beam w=10 | 5-gram | +0.01–0.02 |

## 6.3 — Phoneme Language Model

Create `src/utils/phoneme_lm.py`:
- Train a character-level n-gram LM on canonical phoneme sequences from training data
- Use KenLM or simple Python implementation
- This captures legal phoneme transitions in Arabic/Qur'anic text
- Penalizes illegal phoneme sequences during decoding

## 6.4 — Model Ensemble (Optional)

If multiple strong models exist (e.g., best xls-r-1b and best whisper-large-v3):

Create `src/utils/ensemble.py`:
- **Logit averaging**: average CTC log-probabilities from N models, then decode
  - Requires time-aligning outputs (all models use 16kHz input, but frame rates may differ)
  - If frame rates differ, interpolate to common length
- **Prediction voting**: decode each model independently, then vote on phoneme sequence
  - Use ROVER (Recognizer Output Voting Error Reduction) or majority alignment

Ensemble runs:
| Run | Models | Method | Expected impact |
|-----|--------|--------|----------------|
| single | best model | — | baseline |
| ens_2  | xls-r-1b + whisper-v3 | logit avg | +0.01–0.02 |
| ens_vote | xls-r-1b + whisper-v3 | ROVER | +0.01–0.03 |

## 6.5 — Post-Processing

Create `src/utils/postprocess.py`:
- **Phoneme sequence cleanup**: remove obviously invalid sequences (e.g., triple consonant clusters)
- **Confidence-based filtering**: for each predicted phoneme, compute frame-level CTC posterior probability
  - If confidence < threshold → flag as uncertain → optionally fall back to canonical
  - Tune threshold on dev set to optimize precision/recall tradeoff
- **Minimum duration filtering**: reject phoneme predictions shorter than N frames

## 6.6 — Submission Pipeline

Create `scripts/submit.py`:
- Load best checkpoint (or ensemble)
- Run inference on test set
- Apply beam search decoding
- Apply post-processing
- Align predictions with canonical reference
- Output CSV in IqraEval submission format: `file_id, predicted_phonemes`
- Run evaluation metrics and print summary
- Save submission file

## 6.7 — Final Ablation Summary

Generate `results/final_summary.csv` with ALL runs across all steps:

| Step | Run | Backbone | Frozen | Stages | Data | Decoding | F1 | Prec | Rec |
|------|-----|----------|--------|--------|------|----------|-----|------|-----|
| 2    | 2a  | wav2vec2-base | Y | 1 | train | greedy | ... | ... | ... |
| ...  | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 6    | best | ... | N | 3 | all | beam+LM | ... | ... | ... |

## 6.8 — Paper-Ready Artifacts

Prepare for submission write-up:
- `results/final_summary.csv` — all ablation results
- `results/per_phoneme_accuracy.csv` — which phonemes are hardest
- `results/error_analysis.txt` — common failure modes (FA vs FR patterns)
- Best checkpoint saved for reproducibility

## Success Criteria
- [ ] Best single model trained with combined config
- [ ] Beam search decoding implemented and tuned
- [ ] F1 exceeds BAIC's 0.4726 (stretch goal: >0.50)
- [ ] Submission CSV generated in correct format
- [ ] Full ablation table across all steps compiled

## Output Files
```
src/utils/decoding.py           ← beam search + LM integration
src/utils/phoneme_lm.py         ← n-gram phoneme LM
src/utils/ensemble.py           ← model ensembling
src/utils/postprocess.py        ← prediction cleanup
configs/step6_best_single.yaml
scripts/submit.py
outputs/step6_*/
results/final_summary.csv
results/per_phoneme_accuracy.csv
results/error_analysis.txt
```
