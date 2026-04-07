# Step 5 — Synthetic Mispronunciation Augmentation

## Objective
Generate and incorporate synthetic mispronunciation data to improve error detection. The IqraEval organizers provided TTS augmentation data, and the top teams showed that extra synthetic data was the single most impactful addition. Target: F1 > 0.48.

## Prerequisites
- Step 4 complete (multi-stage pipeline working)
- Best multi-stage checkpoint available

## 5.1 — Use IqraEval TTS Data

The organizers already provide `IqraEval/Iqra_TTS`:
- ~26 hours correct pronunciation (canonical)
- ~26 hours with systematic mispronunciations
- 7 TTS voices (5 male, 2 female)
- Errors based on confusion-pairs matrix (phoneme similarity)

Create `src/data/tts_dataset.py`:
1. Load `IqraEval/Iqra_TTS`
2. Parse both correct and mispronounced splits
3. For mispronounced samples: labels = actual (erroneous) phoneme sequence
4. For correct samples: labels = canonical phoneme sequence
5. Merge with IqraEval training data

## 5.2 — Custom Synthetic Error Generation

Generate additional synthetic errors beyond what IqraEval provides:

Create `src/data/synthetic_errors.py`:

### Error types (applied to canonical phoneme sequences):
1. **Substitution** (most common in Qur'anic recitation)
   - Replace phoneme with acoustically similar one
   - Use confusion matrix from IqraEval phoneme similarity data
   - Error rate: 5–15% of phonemes per utterance
   
2. **Deletion**
   - Drop a phoneme (simulates skipping/swallowing sounds)
   - Error rate: 2–5% of phonemes
   
3. **Insertion**
   - Add an extra phoneme (simulates over-articulation)
   - Error rate: 1–3% of phonemes

4. **Duration errors** (optional, for TTS-generated audio)
   - Shorten/lengthen vowels (violates elongation rules)
   - Implemented as time-stretching segments of audio

### Implementation:
- Take correct audio + canonical phoneme sequence
- Randomly apply 1–2 error types per utterance
- Modified phoneme sequence becomes the label (what was actually pronounced)
- The canonical sequence stays as reference
- This naturally creates (canonical, verbatim, audio) triplets

## 5.3 — Data Mixing Strategy

Control ratio of real vs synthetic data:

| Mix | Real IqraEval | TTS Correct | TTS Errors | Custom Synth |
|-----|--------------|-------------|------------|--------------|
| A   | 100%         | 0%          | 0%         | 0%           |
| B   | 100%         | 100%        | 100%       | 0%           |
| C   | 100%         | 100%        | 100%       | 50%          |
| D   | 100%         | 50%         | 100%       | 100%         |

Use weighted sampling if mixing: upweight real data to prevent synthetic data from dominating.

Implement in DataLoader with `WeightedRandomSampler` or epoch-level data mixing.

## 5.4 — Training with Augmented Data

Use the best multi-stage pipeline from step 4, but in stage 3 (IqraEval fine-tune) add the synthetic data:

Config `configs/step5_augmented.yaml`:
```yaml
# Inherits from step4_stage3.yaml
train_datasets:
  - path: "data/raw/Iqra_train"
    weight: 1.0
  - path: "data/raw/Iqra_TTS"
    weight: 0.5        # down-weight synthetic
  - path: "data/processed/synthetic_errors"
    weight: 0.3        # down-weight custom errors more

encoder_lr: 5e-6
head_lr: 5e-5
max_epochs: 10
```

## 5.5 — Audio-Level Augmentation (on top of SpecAugment)

Add to `src/data/augmentation.py`:
- **Speed perturbation**: factor in [0.9, 1.1] — makes model robust to speaking rate
- **Noise injection**: add low-level white/pink noise (SNR 30–40dB) — mild robustness
- **Pitch shift**: ±1 semitone — speaker variability

Apply to real data only (TTS data already has speaker variety).

## 5.6 — Ablation Runs

| Run | Base model | Extra data | Expected F1 |
|-----|-----------|------------|-------------|
| 5a  | Step 4 best | none (= step 4) | baseline |
| 5b  | Step 4 best | + TTS correct only | +0.00–0.01 |
| 5c  | Step 4 best | + TTS correct + errors | +0.01–0.03 |
| 5d  | Step 4 best | + TTS + custom synth | +0.02–0.04 |
| 5e  | Step 4 best | + all + audio augment | +0.02–0.05 |

## Success Criteria
- [ ] TTS data loads and integrates with training pipeline
- [ ] Custom synthetic error generation produces valid (audio, phoneme) pairs
- [ ] F1 with augmented data > F1 without (from step 4)
- [ ] Data mixing ratio ablation completed

## Output Files
```
src/data/tts_dataset.py
src/data/synthetic_errors.py
configs/step5_augmented.yaml
data/processed/synthetic_errors/  ← generated error data
outputs/step5_*/
results/step5_ablation.csv
```
