# Plans Light — Step 3: Zero-Training MDD via Representation Mismatch (Multi-Layer Anchor Distance)

## Objective
Detect mispronunciation by **comparing the input’s segment embedding to a canonical reference embedding**, purely with a frozen encoder and the anchor bank from Step 1.  
Still **no training** in this step (one small exception allowed: a *closed-form* Mahalanobis calibration using the already-estimated `cov_diag`).

## Motivation
Top systems in [`papers/iqraeval.txt`](../papers/iqraeval.txt) all train a phoneme classifier and inherit Arabic-specific design choices. A clean deep-learning angle — “mispronunciation = geometric mismatch against a reference anchor” — is under-represented and directly publishable as a representation-analysis study.

## Prerequisites
- `plans_light/step1.md` (anchor bank) ready.
- Same frozen encoder used in Step 1.

## 3.1 — Method
For each test utterance and each canonical phone position `i`:

1. Get the aligned segment `[t_s, t_e]` (reuse Step 2’s forced-alignment path or Step 1’s uniform fallback).
2. Extract segment embedding `e_i = mean_t feat[t]` at **multiple layers**:
   - Early layer (acoustic): e.g. layer 4.
   - Mid layer: e.g. layer 12.
   - Late layer (linguistic): last hidden state.
3. Look up anchor `μ_ri` (mean embedding for canonical phone `r_i`) and `Σ_ri` (diagonal covariance) at the same layer.
4. Compute per-layer distances:
   - **Cosine:** `d_cos = 1 − cos(e_i, μ_ri)`.
   - **Mahalanobis (diagonal):** `d_mah = sqrt( Σ_d (e_i,d − μ_ri,d)^2 / (σ_ri,d^2 + ε) )`.
5. Multi-layer fusion (closed-form, no gradient training):
   - `d_fused = w_acoustic * d_early + w_linguistic * d_late`.
   - `w` grid-searched on **dev** only; report the picked value.

## 3.2 — Decision
- `r_i` accepted iff `d_fused < τ` (percentile-threshold on dev).
- Same rejection diagnostic slot as Step 2 (argmax phone at segment center) so the evaluator contract stays identical.

## 3.3 — Ablations
| Ablation | Setting |
|----------|---------|
| B1 | cosine, last layer only |
| B2 | cosine, early + late, fused |
| B3 | Mahalanobis, last layer |
| B4 | Mahalanobis, early + late |
| B5 | best config + uniform segmentation (no CTC alignment) — robustness check |
| B6 | anchor bank estimated on dev only (no test leakage) |

## 3.4 — Combine with Step 2 (optional, still no training)
Score fusion:
```
s_i = γ * (-c_i)   +   (1-γ) * d_fused
```
where `c_i` is Step 2 confidence and `γ ∈ {0, 0.25, 0.5, 0.75, 1}`, picked on dev.  
This gives a **zero-training “ensemble”** result directly comparable to a single trained baseline.

## Success Criteria
- [ ] Zero trainable parameters anywhere.
- [ ] Each ablation B1–B6 produces a metrics row.
- [ ] Step 2 + Step 3 fusion ≥ best single zero-training variant.
- [ ] All numbers reported on released GT with `scripts/evaluate_predictions.py`.

## Output Files
```
src/light/anchor_distance.py
scripts/light_predict_step3.py
scripts/light_fuse_step2_step3.py
outputs/light/step3/predictions.csv
outputs/light/step3/metrics.txt
outputs/light/step3/ablation.csv
outputs/light/fusion/predictions.csv
outputs/light/fusion/metrics.txt
```
