# iqra — Retrieval-based Mispronunciation Detection \& Diagnosis (MDD) for IqraEval

This repository implements a **retrieval-based phoneme-level Mispronunciation Detection and Diagnosis (MDD)** pipeline for Quranic/MSA-style recitation, aligned with the IqraEval benchmark line.

The core idea is simple and inspectable:

- Build a **memory bank** of *canonical (correct)* phoneme segments.
- Embed each segment with a pretrained speech encoder.
- For each query phoneme segment, retrieve top-\(k\) nearest bank segments and **vote**.
- Optionally apply a **lightweight rule-tuning layer** that learns **pair-wise vote weights** for \((\text{expected phoneme}, \text{neighbor label})\) to reduce systematic confusions (without retraining the acoustic encoder).

The accompanying paper draft lives in `plans_mdd/acl.tex`.

## Repository layout (main MDD scripts)

All MDD-related scripts are in `scripts_mdd/`:

- `00_combine_iqraeval_datasets.py`: export unified audio + `metadata.csv` to `data/unified/`
- `01_build_mdd_bank_manifest.py`: create `train/dev/bank` manifests from unified metadata
- `01b_build_quranmb_v2_test_manifest.py`: build a unified-style **test** manifest from QuranMB.v2 + gold labels
- `02_align_and_segment.py`: align utterances to target phonemes and export phoneme segments
- `03_build_embedding_bank.py`: compute one embedding vector per phoneme segment
- `05_retrieve_mdd.py`: kNN retrieval (base or tuned via `--vote-weights`) and metric reporting
- `04_train_vote_weights.py`: train pair-wise vote weights (rule tuning) using train/dev splits
- `06_analyze_tuning_pair_improvements.py`: analyze which \((\text{expected},\text{majority})\) pairs improved after tuning

## Setup

### Python dependencies

Install the Python requirements:

```bash
pip install -r requirements.txt
```

Retrieval uses FAISS; install one of:

```bash
pip install faiss-cpu
# or (if your environment supports it)
pip install faiss-gpu
```

### Hugging Face access

Some datasets/models are pulled from the Hugging Face Hub. Set a token if required:

```bash
export HF_TOKEN="..."
```

For QuranMB.v2 test + gold labels you may need to accept dataset terms on the Hub.

## Data pipeline (end-to-end)

### 1) Build a unified local corpus

Exports wav files and a unified `metadata.csv`:

```bash
python scripts_mdd/00_combine_iqraeval_datasets.py
```

Outputs:

- `data/unified/wav/*.wav`
- `data/unified/metadata.csv`
- `data/unified/stats.json`

### 2) Split into train / dev and build a correct-only bank

```bash
python scripts_mdd/01_build_mdd_bank_manifest.py
```

Outputs:

- `data/unified/train/manifest.csv`
- `data/unified/dev/manifest.csv`
- `data/unified/bank/manifest.csv` (correct-only subset)

### 3) Build the official test manifest (QuranMB.v2)

```bash
python scripts_mdd/01b_build_quranmb_v2_test_manifest.py
```

Outputs:

- `data/unified/test/manifest.csv` (plus metadata files under `data/unified/test/`)

### 4) Align and segment into phoneme-level chunks

Run alignment + segmentation for bank/train/dev/test. The CTC backend is recommended when the model vocabulary matches your phone set; otherwise, uniform segmentation is available.

Examples:

```bash
# bank/train/dev presets
python scripts_mdd/02_align_and_segment.py --dataset-set bank --backend ctc_forced
python scripts_mdd/02_align_and_segment.py --dataset-set train --backend ctc_forced
python scripts_mdd/02_align_and_segment.py --dataset-set dev --backend ctc_forced

# test uses a custom manifest/output dir
python scripts_mdd/02_align_and_segment.py \
  --dataset-set custom \
  --input-manifest data/unified/test/manifest.csv \
  --output-dir data/unified/test/segments \
  --backend ctc_forced
```

### 5) Build phoneme segment embeddings

```bash
python scripts_mdd/03_build_embedding_bank.py --dataset-set bank
python scripts_mdd/03_build_embedding_bank.py --dataset-set train
python scripts_mdd/03_build_embedding_bank.py --dataset-set dev
python scripts_mdd/03_build_embedding_bank.py --dataset-set custom \
  --segments-csv data/unified/test/segments/bank_segments.csv \
  --output-dir data/unified/test/embeddings
```

### 6) Run retrieval (base model)

Test-time retrieval writes one folder per `k`:

```bash
python scripts_mdd/05_retrieve_mdd.py --query-set test --top-k 8
python scripts_mdd/05_retrieve_mdd.py --query-set test --top-k 16
```

Outputs:

- `data/unified/test/retrieval_<k>/mdd_retrieval_results.csv`
- `data/unified/test/retrieval_<k>/mdd_retrieval_stats.json`

### 7) Train rule-tuning weights (pair-wise vote calibration)

This learns weights \(w_{e,y}\) over \((\text{expected phoneme}, \text{neighbor label})\) pairs by iteratively down-weighting false-reject-prone pairs on train and monitoring dev.

```bash
python scripts_mdd/04_train_vote_weights.py --top-k 16 --output-dir data/unified/train/vote_train_16
```

Key outputs:

- `data/unified/train/vote_train_16/vote_weights.json`
- `data/unified/train/vote_train_16/vote_weights.best_dev.json`

### 8) Run retrieval with tuned weights

```bash
python scripts_mdd/05_retrieve_mdd.py --query-set test --top-k 16 \
  --vote-weights data/unified/train/vote_train_16/vote_weights.best_dev.json
```

### 9) Analyze which phoneme pairs improved

Compare base vs tuned pair tables and write per-pair reductions (count and %):

```bash
python scripts_mdd/06_analyze_tuning_pair_improvements.py \
  --test-root data/unified/test \
  --ks 8 16 \
  --table-key fr_rows_gold_ok_pred_wrong
```

Outputs:

- `data/unified/test/retrieval_<k>/pair_reduction_fr_rows_gold_ok_pred_wrong.csv`
- `data/unified/test/pair_reduction_summary_fr_rows_gold_ok_pred_wrong.json`

## Notes / common issues

- **FAISS required**: `scripts_mdd/05_retrieve_mdd.py` requires FAISS; install `faiss-cpu` or `faiss-gpu`.
- **Caching**: retrieval scripts cache nearest-neighbor indices under each `retrieval_<k>/` folder to avoid recomputing.
- **CTC alignment vocab**: if forced alignment fails due to phone/token mismatch, the segmenter can fall back to uniform segmentation.