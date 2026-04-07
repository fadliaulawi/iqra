# Step 4 — Multi-Stage Domain Adaptation

## Objective
Instead of fine-tuning directly on IqraEval, adapt the encoder progressively: general Arabic speech → Qur'anic recitation → IqraEval phoneme task. This mirrors the Metapseud approach (multi-stage fine-tuning) which showed strong results. Target: F1 > 0.47.

## Prerequisites
- Step 3 complete (full fine-tuning pipeline working)
- Best backbone identified (likely xls-r-1b or whisper-large-v3)

## 4.1 — Data for Each Stage

### Stage 1: General Arabic Speech
- Dataset: `IqraEval/Iqra_train` (the CV-Ar subset, ~82h)
- Task: CTC phoneme recognition on general Arabic read speech
- Purpose: adapt encoder to Arabic phonetics without overfitting to Qur'anic patterns

### Stage 2: Qur'anic Recitation
- Dataset: EveryAyah/QUL subset (following Hafs2Vec approach)
  - Download: professional Qur'anic recitation audio
  - ~94 hours from 28 reciters (filter to clips <10s → ~54k clips)
  - Generate phoneme labels using the IqraEval phonemizer
- Alternative: use the Qur'anic-augmented portion of the CV-Ar data if EveryAyah is inaccessible
- Task: CTC phoneme recognition on correct Qur'anic recitation
- Purpose: shift encoder representations toward Qur'anic articulation patterns

### Stage 3: IqraEval Fine-tune
- Dataset: `IqraEval/Iqra_train` (full, including Qur'anic samples)
- Task: CTC phoneme recognition — same as step 3 but starting from stage 2 checkpoint
- Purpose: final task-specific adaptation

## 4.2 — Training Schedule

Each stage uses decreasing learning rate and fewer epochs:

| Stage | Data | Encoder LR | Head LR | Epochs | Warmup |
|-------|------|-----------|---------|--------|--------|
| S1    | CV-Ar general | 3e-5 | 3e-4 | 5 | 10% |
| S2    | Qur'anic recitation | 1e-5 | 1e-4 | 5 | 10% |
| S3    | IqraEval train | 5e-6 | 5e-5 | 10 | 5% |

Rationale: each stage narrows the domain, so we reduce LR to avoid forgetting.

## 4.3 — EveryAyah Data Pipeline

Create `src/data/everyayah.py`:
1. Download audio from EveryAyah API or mirror
2. Filter: keep clips < 10 seconds
3. Generate phoneme labels:
   - Get Qur'anic text for each verse
   - Run through IqraEval phonemizer to get canonical phoneme sequence
   - These are correct recitations → labels = canonical phonemes
4. Save in same format as IqraEval training data

## 4.4 — Stage Management

Update `src/training/train.py` to support:
- `--resume_from` flag: load checkpoint from previous stage
- `--stage` flag: select config section
- When resuming, reinitialize optimizer and scheduler (new stage = fresh optimizer state)

Alternatively, create `scripts/run_multistage.sh`:
```bash
# Stage 1
python src/training/train.py --config configs/step4_stage1.yaml

# Stage 2 (resume from stage 1 best)
python src/training/train.py --config configs/step4_stage2.yaml \
    --resume_from outputs/step4_stage1/best_checkpoint.pt

# Stage 3 (resume from stage 2 best)
python src/training/train.py --config configs/step4_stage3.yaml \
    --resume_from outputs/step4_stage2/best_checkpoint.pt
```

## 4.5 — Ablation Runs

| Run | Stages | Backbone | Expected F1 |
|-----|--------|----------|-------------|
| 4a  | S3 only (= step 3 best) | xls-r-1b | baseline |
| 4b  | S1 → S3 (skip Qur'anic) | xls-r-1b | +0.01–0.02 |
| 4c  | S2 → S3 (skip general Arabic) | xls-r-1b | +0.01–0.02 |
| 4d  | S1 → S2 → S3 (full pipeline) | xls-r-1b | +0.02–0.04 |
| 4e  | S1 → S2 → S3 | whisper-large-v3 | compare |

Key comparison: 4a vs 4d tells you if multi-stage is worth the complexity.

## 4.6 — Monitoring

Track per-stage:
- Training loss curve
- F1 on open_testset (evaluated at end of each stage)
- Per-phoneme accuracy heatmap (identify which phonemes improve with domain adaptation)

## Success Criteria
- [ ] Multi-stage pipeline runs end-to-end (S1 → S2 → S3)
- [ ] F1 of S1→S2→S3 > F1 of S3-only (from step 3)
- [ ] Qur'anic recitation data successfully loaded and phonemized
- [ ] Ablation table shows clear stage contribution

## Output Files
```
src/data/everyayah.py
configs/step4_stage1.yaml
configs/step4_stage2.yaml
configs/step4_stage3.yaml
scripts/run_multistage.sh
outputs/step4_*/
results/step4_ablation.csv
```
