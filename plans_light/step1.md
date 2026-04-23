# Plans Light — Step 1: Infrastructure Reuse + Reference Anchor Bank

## Objective
Set up the **light-research track** by reusing the existing data pipeline, vocab, and evaluation scripts, and build a one-time **per-canonical-phoneme reference embedding bank** from the training data.  
**No model is trained in this step.** The goal is to have everything needed for zero-training MDD in Step 2 and Step 3.

## Prerequisites
- Main-track Step 1 complete (`data/raw/train/`, `data/raw/dev/`, `data/raw/test/`, `data/raw/test_gt/`, `data/phoneme_vocab.json`).
- `src/evaluation/metrics.py` and `scripts/evaluate_predictions.py` working.

## 1.1 — Directory Layout
```
src/light/
  __init__.py
  embeddings.py        # shared feature extraction (Wav2Vec2 / XLS-R)
  anchor_bank.py       # build / load per-phoneme reference anchors
scripts/
  light_build_anchor_bank.py
  light_predict_*.py   # added later in Step 2 / 3
outputs/light/
  anchor_bank.pt
```

## 1.2 — Shared Encoder Policy
Pick **one frozen SSL encoder** and reuse it across all light-track steps:
- Default: `facebook/wav2vec2-xls-r-300m` (cheap, multilingual, strong).
- Optional alt: `facebook/mms-1b-all` with `target_lang=ara` (for ablation only).

Expose in `src/light/embeddings.py`:
- `load_encoder(name) -> (model, processor)` — frozen, eval-mode, no dropout.
- `extract_frame_features(wav, model, processor, layer="weighted") -> Tensor[T, D]`:
  - Support **last hidden state**, a **specific layer index**, or a **static uniform average** of all layers (no learned weights — keep zero-training).
- `extract_forced_alignment(wav, ref_phones) -> List[(phone, start_frame, end_frame)]`:
  - Use `torchaudio` forced alignment or CTC posterior peaks from a pretrained phoneme head.
  - If no phoneme head is trivially available, fall back to **uniform segmentation** by reference phone count (clearly documented limitation).

## 1.3 — Build the Reference Anchor Bank
`scripts/light_build_anchor_bank.py`:
- Iterate over **correct** training utterances (use `Reference_phn` from `data/raw/train/metadata.csv`).
- For each utterance:
  - Extract frame features.
  - Get a **phone-level segmentation** (Section 1.2).
  - Average the frames in each segment to get one vector per phone occurrence.
- Aggregate across occurrences:
  - For each canonical phone `p`, compute:
    - `mean_p = mean of its segment vectors`
    - `cov_p = diagonal covariance (numerical stability only)`
    - `n_p = count of occurrences used`
- Save as `outputs/light/anchor_bank.pt`:
  ```python
  {
      "encoder_name": str,
      "phoneme_vocab": dict,
      "mean": Tensor[P, D],
      "cov_diag": Tensor[P, D],
      "count": Tensor[P],
  }
  ```

**Notes:**
- Cap segments per phone (e.g. max 5k) to bound compute and speaker skew.
- Write a tiny `anchor_bank.json` summary (counts, missing phones) next to the `.pt`.

## 1.4 — Sanity Checks
Add `scripts/light_sanity.py`:
- Load anchor bank.
- For 20 random training utterances, compute per-phone cosine similarity between the utterance segment embedding and the matching anchor.
- Print mean/median similarity and any phone with `< 0.3` average (possible bad segmentation).

## Success Criteria
- [ ] `outputs/light/anchor_bank.pt` built with ≥ 60 out of 62 canonical phones represented.
- [ ] Sanity script runs end-to-end; per-phone mean similarity on correct utterances is clearly higher than on random-pair utterances (~+0.1 absolute minimum).
- [ ] No training has been performed.

## Output Files
```
src/light/embeddings.py
src/light/anchor_bank.py
scripts/light_build_anchor_bank.py
scripts/light_sanity.py
outputs/light/anchor_bank.pt
outputs/light/anchor_bank.json
```
