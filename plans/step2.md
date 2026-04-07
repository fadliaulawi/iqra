# Step 2 — Frozen SSL Encoder + CTC Head (Minimal Baseline)

## Objective
Train the simplest possible working model: frozen pretrained encoder + lightweight CTC head. This establishes the training loop, checkpointing, inference, and end-to-end evaluation. Expect results near or below the official baselines (F1 ~0.35–0.44).

## Prerequisites
- Step 1 complete (data loaded, vocab built, eval script working)

## 2.1 — Model Architecture

Create `src/models/ctc_model.py`:

```
FrozenSSL_CTC:
    encoder:    pretrained SSL model (FROZEN, no grad)
    weighted_sum: learnable scalar weights per transformer layer (following SUPERB protocol)
    head:       2-layer BiLSTM (hidden=1024) → Linear(hidden, num_phonemes)
    loss:       CTC loss
```

Details:
- **Encoder**: load from HuggingFace, freeze all parameters
- **Weighted sum**: `nn.Parameter(torch.ones(num_layers))` → softmax → weighted sum of all hidden states
  - Requires `output_hidden_states=True` from the encoder
- **BiLSTM head**: input_dim = encoder_hidden_dim, hidden = 1024, layers = 2, dropout = 0.1
- **Output**: Linear projection to `len(phoneme_vocab)` (including blank)
- **CTC Loss**: `torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)`

## 2.2 — Training Configuration

Create `configs/step2_frozen_baseline.yaml`:

```yaml
# --- Model ---
encoder_name: "facebook/wav2vec2-xls-r-300m"  # start small, ablate later
freeze_encoder: true
head_type: "bilstm"
head_hidden: 1024
head_layers: 2
head_dropout: 0.1
num_phonemes: 65  # 62 phonemes + pad + unk + blank

# --- Data ---
train_dataset: "data/raw/Iqra_train"
eval_dataset: "data/raw/open_testset"
sample_rate: 16000
max_audio_len_sec: 15.0  # drop longer clips

# --- Training ---
batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 3e-4
weight_decay: 0.01
optimizer: "adamw"
scheduler: "cosine"
warmup_steps: 500
max_epochs: 15
fp16: true

# --- Eval ---
eval_every_epoch: 1
save_best_metric: "f1"
patience: 5  # early stopping

# --- Decoding ---
decoding: "greedy"  # greedy CTC decoding for now

# --- Output ---
output_dir: "outputs/step2_frozen_xls-r-300m"
```

## 2.3 — Training Script

Create `src/training/train.py`:

Core loop:
1. Load config from yaml
2. Build dataset + dataloader (from step 1)
3. Build model (from 2.1)
4. Optimizer: AdamW on non-frozen params only
5. Training loop:
   - Forward pass → CTC loss
   - Backward + gradient clipping (max_norm=5.0)
   - Log loss to tensorboard every 50 steps
6. Evaluation every epoch:
   - Greedy CTC decode: `argmax → collapse repeats → remove blanks`
   - Run evaluation metrics (from step 1)
   - Log all 9 metrics
   - Save checkpoint if F1 improves
7. Early stopping after `patience` epochs with no F1 improvement

## 2.4 — Inference Script

Create `src/training/predict.py`:
- Load trained checkpoint
- Run inference on test set
- Greedy CTC decode
- Save predictions as CSV: `file_id, predicted_phonemes` (one row per utterance)
- Run evaluation and print metrics

## 2.5 — Ablation Runs (within this step)

Run these variants to understand encoder impact:

| Run | Encoder | Params | Expected VRAM |
|-----|---------|--------|---------------|
| 2a  | `facebook/wav2vec2-base` | 95M | ~8GB |
| 2b  | `facebook/wav2vec2-xls-r-300m` | 300M | ~16GB |
| 2c  | `facebook/wav2vec2-xls-r-1b` | 1B | ~25GB |
| 2d  | `microsoft/wavlm-base` | 95M | ~8GB |

All frozen, same head, same hyperparams. Log results in a comparison table.

## 2.6 — CTC Greedy Decoding

Create `src/utils/decoding.py`:
- `greedy_decode(logits) -> phoneme_ids`:
  - argmax over vocab dim at each frame
  - collapse consecutive duplicates
  - remove blank tokens
- `decode_to_str(phoneme_ids, vocab) -> str`: map ids back to phoneme strings

## Success Criteria
- [ ] Training runs to completion without OOM
- [ ] Loss decreases over epochs
- [ ] F1 on open_testset is in range 0.30–0.44 (near official baselines)
- [ ] Predictions CSV is generated and passes evaluation
- [ ] Ablation table across 4 encoders is filled

## Output Files
```
src/models/ctc_model.py
src/training/train.py
src/training/predict.py
src/utils/decoding.py
configs/step2_frozen_baseline.yaml
outputs/step2_*/                    ← checkpoints + logs per run
results/step2_ablation.csv          ← encoder comparison table
```
