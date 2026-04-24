# Step 3.5 — Tajweed-Native Training vs Iqra-Only Training (Balance-Oriented MDD)

## Objective
Test whether **Tajweed-native phoneme supervision** improves the **precision-recall balance** of MDD compared with standard Iqra-only phoneme supervision.

This step is **not leaderboard-oriented**. The primary goal is to improve operating behavior (especially reducing false alarms) rather than matching official benchmark comparability.

- **Run A (Iqra-only):** train with current Iqra phoneme labels from `data/raw/{train,dev}/metadata.csv`.
- **Run B (Tajweed-native):** train with Tajweed phoneme labels generated directly from Qur'anic text via `quranic-phonemizer`, without collapsing to Iqra labels for training.
- **Claim:** richer Tajweed label space can move the MDD operating point toward better balance (higher precision at usable recall, improved FAR/FRR trade-off).

## Scope and Positioning

1. This is a **new task setting** with a different phoneme ontology (Tajweed-native).
2. Main comparison is **A vs B under identical model/training budget**.
3. If Iqra-space metrics are reported for B, they are **secondary diagnostics** and must be clearly labeled as non-leaderboard.

## Prerequisites

- Step 3 training and prediction pipeline works (`src/training/train.py`, `src/training/predict.py`).
- Tajweed G2P dependency installed (`quranic-phonemizer`).
- Current best config available as template (e.g. `configs/step3_finetune_xls_r_1b.yaml`).
- Text source for each train/dev sample can be resolved (direct column or HF join by `id`).

## 3.5.1 — Version Tajweed labels (reproducibility first)

1. Pin `quranic-phonemizer` version (and commit if installed from source) in:
   - `requirements_tajweed.txt`
   - `data/processed/tajweed_version.lock`
2. Keep source phoneme inventory reference in:
   - `data/processed/TAJWEED_VERSION.md`
3. Save a small reproducibility note:
   - phonemizer version
   - options used (stop signs, separators, etc.)
   - whether post-processing normalization is applied

## 3.5.2 — Build parallel Tajweed metadata (no in-place edits)

1. Copy official metadata:
   - `data/raw/train/metadata.csv` -> `data/raw/train_tajweed_native/metadata.csv`
   - `data/raw/dev/metadata.csv` -> `data/raw/dev_tajweed_native/metadata.csv`
2. Resolve Qur'anic text per row (metadata field or HF join by `id`).
3. For each row:
   - `phones_tajweed` = `Phonemizer().phonemize(text).phonemes_str(...)`
   - `phonemes`, `phoneme_ref`, `phoneme_aug` = Tajweed-native sequence (no Iqra collapse for training).
4. Keep `ID` and `wav_path` unchanged.
5. Recompute `duration_s` from wav metadata.
6. Drop rows with missing text or empty label after normalization; log counts.

## 3.5.3 — Build Tajweed vocab and tokenizer path

1. Build Tajweed vocabulary from train_tajweed_native labels:
   - include required specials (`<pad>`, `<unk>`, `<blank>`).
2. Save vocab as:
   - `data/phoneme_vocab_tajweed.json`
3. Verify:
   - dev token coverage against train vocab
   - OOV rate on dev (`<unk>` usage)
4. Keep original Iqra vocab untouched (`data/phoneme_vocab.json`).

## 3.5.4 — Add frozen configs for A/B

Create two configs with identical hyperparameters except data/vocab/output paths:

| Run | `train_csv` | `val_csv` | `vocab_path` | `output_dir` |
|-----|-------------|-----------|--------------|--------------|
| **A (Iqra-only)** | `data/raw/train/metadata.csv` | `data/raw/dev/metadata.csv` | `data/phoneme_vocab.json` | `outputs/step3.5_iqra_xls_r_1b` |
| **B (Tajweed-native)** | `data/raw/train_tajweed_native/metadata.csv` | `data/raw/dev_tajweed_native/metadata.csv` | `data/phoneme_vocab_tajweed.json` | `outputs/step3.5_tajweed_native_xls_r_1b` |

No weighted sampling, no curriculum changes, no loss re-weighting in this step.

## 3.5.5 — Train and predict (same budget)

1. `python src/training/train.py --config configs/step3.5_iqra_xls_r_1b.yaml`
2. `python src/training/train.py --config configs/step3.5_tajweed_native_xls_r_1b.yaml`
3. Run prediction for each using its own config.

## 3.5.6 — Evaluate for balance (primary objective)

Primary report should emphasize operating-point balance:

1. Precision, Recall, F1
2. FAR, FRR
3. Balance indicator:
   - `abs(Precision - Recall)` (lower is better)
   - plus threshold where FAR approximately FRR (EER-like operating point)
4. Save summary table:
   - `results/step3.5_balance.csv`

Optional diagnostic:
- If converting B outputs to Iqra space for comparison, mark clearly as **diagnostic only**.

## 3.5.7 — How to write results in paper

- Focus question: does Tajweed-native training produce a better practical trade-off?
- If B improves precision with acceptable recall drop: argue reduction in false alarms.
- If B improves both precision and recall: strong evidence for better label inductive bias.
- If mixed: report subgroup/phoneme-level behavior (vowels, emphatics, gemination-heavy cases).
- If no gain: report honestly and discuss domain mismatch or label noise.

## 3.5.8 — What this step is *not*

- Not an official leaderboard submission setup.
- Not a claim that Tajweed and Iqra phoneme spaces are equivalent.
- Not dependent on a manual one-to-one Tajweed->Iqra mapping for the primary experiment.

## Success criteria

- [ ] A and B share architecture and training budget; only label space/vocab differ.
- [ ] Tajweed-native metadata and vocab are reproducible from pinned tooling.
- [ ] Results include PR and FAR/FRR trade-off (not only a single headline score).
- [ ] Paper narrative is explicit that this is a balance-oriented, non-leaderboard objective.

## Expected outputs

```
data/raw/train_tajweed_native/metadata.csv
data/raw/dev_tajweed_native/metadata.csv
data/phoneme_vocab_tajweed.json
configs/step3.5_iqra_xls_r_1b.yaml
configs/step3.5_tajweed_native_xls_r_1b.yaml
outputs/step3.5_iqra_xls_r_1b/
outputs/step3.5_tajweed_native_xls_r_1b/
results/step3.5_balance.csv
```
