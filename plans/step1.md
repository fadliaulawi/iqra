# Step 1 — Environment, Data, Project Scaffold, Evaluation Script

## Objective
Set up a fully working project: dependencies installed, all IqraEval data downloaded and explored, phoneme vocabulary built, official evaluation metrics implemented, and a dummy prediction that runs end-to-end through eval. No model training yet.

## 1.1 — Dependencies

Create `requirements.txt`:

```
torch>=2.1
torchaudio>=2.1
transformers>=4.38
datasets>=2.18
accelerate>=0.27
evaluate
soundfile
librosa
pandas
numpy
scipy
tensorboard
pyyaml
```

Install with: `pip install -r requirements.txt`

## 1.2 — Data Download

Download from HuggingFace org `IqraEval`:

| Dataset | HF ID | Purpose |
|---------|--------|---------|
| Training set | `IqraEval/Iqra_Train` | Main training data (~79h, ~74k clips) |
| TTS augmentation | `Yasel/SWS_TTS` (or `IqraEval/Iqra_TTS` if available) | Synthetic correct + mispronounced |
| Open test set | `IqraEval/open_testset` | Public evaluation split |

Note: the old `IqraEval/dummy_samples` dataset is not used (often unavailable); use `scripts/test_pipeline.py` for smoke tests and optionally `--max-samples` on train/test for quick checks.

Code should:
1. Use `datasets.load_dataset()` for each
2. Save to `data/raw/` in a consistent format
3. Print dataset statistics: num samples, total duration, speaker counts, phoneme distribution

## 1.3 — Phoneme Vocabulary

Build phoneme vocabulary from training data:
- Extract all unique phoneme tokens from the training transcripts
- Should result in ~62 unique phonemes (per IqraEval spec)
- Save as `data/phoneme_vocab.json` mapping `{phoneme: id, ...}` with special tokens: `<pad>=0`, `<unk>=1`, `<blank>=2` (for CTC)

## 1.4 — Data Preprocessing Pipeline

Create `src/data/dataset.py`:
- Audio loading: resample all audio to 16kHz mono
- Tokenization: convert phoneme string sequences to integer sequences using vocab
- Return dict: `{"audio": waveform, "audio_len": int, "labels": int_tensor, "label_len": int}`
- Implement a PyTorch Dataset and DataLoader with proper collation (pad audio and labels)

Verify by loading 5 samples, printing shapes and decoded phoneme text.

## 1.5 — Evaluation Script

Create `src/evaluation/metrics.py`:

Implement the official IqraEval evaluation protocol:
1. **Three-way alignment**: align (canonical, verbatim, predicted) phoneme sequences
   - Use edit-distance / dynamic programming alignment
2. **Classification**: categorize each aligned position as TA, TR, FA, FR, CD, ED
3. **Metrics**: compute FRR, FAR, DER, Precision, Recall, F1, Correct Rate, Accuracy
   - FRR = FR / (TA + FR)
   - FAR = FA / (FA + TR)
   - DER = ED / (CD + ED)
   - Precision = TR / (TR + FR)
   - Recall = TR / (TR + FA)
   - F1 = 2 * Precision * Recall / (Precision + Recall)

Also create `src/evaluation/alignment.py`:
- Implement phoneme sequence alignment using edit distance
- Handle insertions, deletions, substitutions
- Return aligned sequences with gap markers

## 1.6 — End-to-End Smoke Test

Create `scripts/test_pipeline.py`:
- Use **synthetic** phoneme sequences (no HF dummy dataset required)
- Generate random phoneme predictions vs reference/verbatim
- Run through evaluation metrics
- Print all metrics and verify the pipeline runs without errors

## Success Criteria
- [ ] All datasets downloaded and loadable
- [ ] Phoneme vocab has ~62 tokens + special tokens
- [ ] DataLoader returns correct shapes
- [ ] Evaluation script produces all 9 metrics (F1, Precision, Recall, FRR, FAR, DER, Correct Rate, Accuracy, CD)
- [ ] Dummy test runs end-to-end in <1 minute

## Output Files
```
data/raw/                     ← downloaded datasets
data/phoneme_vocab.json       ← phoneme-to-id mapping
src/data/dataset.py           ← Dataset + DataLoader
src/evaluation/metrics.py     ← official metrics
src/evaluation/alignment.py   ← sequence alignment
scripts/test_pipeline.py      ← end-to-end smoke test
requirements.txt              ← dependencies
```
