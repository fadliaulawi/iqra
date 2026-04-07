# Step 3 — Full Fine-Tuning, Single Stage

## Objective
Unfreeze the SSL encoder and fine-tune end-to-end on IqraEval training data. This is the single biggest expected accuracy jump. Also introduce the Whisper-large-v3 track as a second backbone. Target: F1 > 0.44 (beat official baselines).

## Prerequisites
- Step 2 complete (training loop, eval, decoding all working)
- Best frozen encoder identified from step 2 ablation

## 3.1 — Model Changes

Modify `src/models/ctc_model.py`:
- Add `freeze_encoder: bool` flag — when False, encoder params get gradients
- Add learning rate groups:
  - encoder params: low LR (1e-5 to 3e-5)
  - head params: higher LR (1e-4 to 3e-4)
- This differential LR prevents catastrophic forgetting of pretrained features

## 3.2 — Wav2Vec2 / XLS-R Track

Config `configs/step3_finetune_xls-r.yaml`:

```yaml
encoder_name: "facebook/wav2vec2-xls-r-1b"  # or best from step2
freeze_encoder: false
head_type: "bilstm"
head_hidden: 1024
head_layers: 2

# Differential LR
encoder_lr: 1e-5
head_lr: 3e-4
weight_decay: 0.01
optimizer: "adamw"
scheduler: "cosine"
warmup_ratio: 0.1

batch_size: 8         # smaller due to full backprop through 1B encoder
gradient_accumulation_steps: 4  # effective batch = 32
max_epochs: 10
fp16: true
gradient_checkpointing: true   # ESSENTIAL for 1B model on 40GB
max_grad_norm: 5.0

eval_every_epoch: 1
save_best_metric: "f1"
patience: 3

output_dir: "outputs/step3_finetune_xls-r-1b"
```

## 3.3 — Whisper-large-v3 Track

Create `src/models/whisper_phoneme.py`:

Whisper is an encoder-decoder model. Two options:

**Option A — Encoder-only + CTC head (recommended for consistency):**
- Extract Whisper encoder only
- Attach same BiLSTM + CTC head as wav2vec2 track
- Fine-tune encoder + head with CTC loss
- Advantage: same decoding pipeline, directly comparable

**Option B — Full encoder-decoder with phoneme tokens:**
- Extend Whisper tokenizer with 62 phoneme tokens (like ANLPers)
- Resize embedding layer
- Fine-tune decoder to produce phoneme sequences
- Decode with autoregressive generation
- Advantage: leverages Whisper's decoder; disadvantage: different pipeline

Implement both, but prioritize Option A for cleaner ablation.

Config `configs/step3_finetune_whisper.yaml`:

```yaml
encoder_name: "openai/whisper-large-v3"
model_type: "whisper_encoder_ctc"  # Option A
freeze_encoder: false
head_type: "bilstm"
head_hidden: 1024
head_layers: 2

encoder_lr: 1e-5
head_lr: 3e-4
batch_size: 8
gradient_accumulation_steps: 4
max_epochs: 10
fp16: true
gradient_checkpointing: true

output_dir: "outputs/step3_finetune_whisper-large-v3"
```

## 3.4 — SpecAugment

Add SpecAugment during training (standard regularization for speech):
- Frequency masking: 2 masks, max width 27
- Time masking: 2 masks, max width 100 frames (or 10% of sequence)

Implement in `src/data/augmentation.py` and apply on-the-fly during training.

## 3.5 — Training Modifications

Update `src/training/train.py`:
- Support differential learning rate groups
- Add gradient checkpointing toggle
- Add SpecAugment toggle
- Add learning rate warmup (warmup_ratio of total steps)

## 3.6 — Ablation Runs

| Run | Encoder | Frozen | SpecAug | Expected F1 |
|-----|---------|--------|---------|-------------|
| 3a  | xls-r-300m | No | No  | ~0.42–0.45 |
| 3b  | xls-r-1b   | No | No  | ~0.44–0.47 |
| 3c  | xls-r-1b   | No | Yes | ~0.45–0.48 |
| 3d  | whisper-large-v3 (enc+CTC) | No | No | ~0.43–0.46 |
| 3e  | whisper-large-v3 (enc+CTC) | No | Yes | ~0.44–0.47 |

## Success Criteria
- [ ] Full fine-tuning runs without OOM (gradient checkpointing enabled)
- [ ] F1 exceeds best frozen baseline from step 2
- [ ] At least one config beats official mHuBERT baseline (F1 > 0.4414)
- [ ] Both wav2vec2 and whisper tracks produce valid results

## Output Files
```
src/models/whisper_phoneme.py
src/data/augmentation.py
configs/step3_finetune_xls-r.yaml
configs/step3_finetune_whisper.yaml
outputs/step3_*/
results/step3_ablation.csv
```
