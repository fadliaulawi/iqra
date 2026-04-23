# Plans Light — Step 4: Minimal Supervised Head (Light Training, Final Ablation)

## Objective
Test whether a **single tiny MLP head** on the diff-feature `[e_i − μ_ri, e_i ⊙ μ_ri, d_cos, d_mah, c_i]` improves over the fully zero-training setup of Steps 2/3.  
This is the **only** plans-light step that involves training, and it is intentionally minimal.

## Constraints (for paper honesty)
- **Frozen encoder** (no gradient through SSL).
- **Classifier size:** ≤ 3 linear layers, ≤ 1M parameters total.
- **Training data:** only the **official IqraEval train split**; no external data.
- **Training budget:** ≤ 1 GPU-hour.
- **Inference budget:** same as Step 3 (negligible overhead).

## 4.1 — Labels
Build a **per-position binary label** automatically:
- For every canonical phone position in the train set, label `1` if `Annotation_phn[i] ≠ Reference_phn[i]`, else `0`.
- No manual annotation needed — reuses the IqraEval training data as-is.

## 4.2 — Features per Position
```
x_i = concat(
  e_i - μ_ri,          # mismatch direction
  e_i * μ_ri,          # gating interaction
  [d_cos, d_mah, c_i]  # scalar zero-training scores
)
```
All components are produced by Steps 1–3 with **no backprop through the encoder**.

## 4.3 — Model
- 3-layer MLP: `D → 256 → 64 → 1`, ReLU, dropout 0.1.
- Sigmoid output = probability of mispronunciation at that position.
- **Loss:** binary cross-entropy with **class weight** to handle imbalance.
- **Optimizer:** AdamW, `lr=1e-3`, `wd=1e-4`.
- **Schedule:** cosine over ≤ 10 epochs, early stop on dev F1.
- **Optional temporal smoothing:** 1D conv or CRF over `{x_i}`. Keep ablatable on/off.

## 4.4 — Decision and Evaluation
- Accept iff `p_i < τ`, with `τ` chosen on dev F1.
- Same diagnostic-slot rule as Steps 2/3 to keep the evaluator interface fixed.

## 4.5 — Ablations
| Ablation | Setting |
|----------|---------|
| C1 | zero-training Step 2+3 fusion (reference, no training) |
| C2 | MLP head on Step 3 features only |
| C3 | MLP head on full concat (C1 features) |
| C4 | C3 + temporal smoothing |
| C5 | C3 with encoder swapped (XLS-R ↔ MMS) — reuse of infrastructure |

## Success Criteria
- [ ] C3 improves over C1 F1 by a meaningful margin.
- [ ] Training time per config ≤ 1 GPU-hour, total pipeline repeatable end-to-end in a day.
- [ ] A final comparison table is produced (all C1–C5, plus numbers from the main plans/ track for context).

## Output Files
```
src/light/supervised_head.py
scripts/light_train_head.py
scripts/light_predict_step4.py
outputs/light/step4/predictions.csv
outputs/light/step4/metrics.txt
outputs/light/step4/ablation.csv
outputs/light/final_table.csv
```
