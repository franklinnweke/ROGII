# Research Notes

## Kaggle Code Observations

Inspected on June 18, 2026 from the ROGII competition Code tab.

High-scoring public notebooks are not just generic tabular regressors. The visible pattern is:

- Hold a strong physical/continuity baseline first.
- Add a second sequence/log-correlation or likelihood-style pipeline.
- Blend pipelines only with guards or self-checking overrides.
- Avoid relying on train-only formation-surface columns because public test files expose only `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input` in horizontal wells, plus `TVT` and `GR` in typewells.

Specific visible references:

- `ROGII Dual Pipeline + Self-Verifying` reported public scores around 7.5 and describes a ridge-style pipeline, a likelihood/particle-filter plus GBM stack, final blending, and guarded physical override.
- `ROGII - Why 3000 Teams Are Stuck at ~12` demonstrates that the evaluation zone is a contiguous tail in every train well, confirms sample-submission row IDs map to hidden `TVT_input` rows, and reports a flat last-known-`TVT_input` baseline around 15.9 RMSE under an honest eval-tail validation.
- A simple LightGBM trained on legal row features can underperform the flat baseline badly when validated on hidden tails, which argues against rushing a tabular model without a strong physical prior.

## Domain/Technical References

- Gamma ray logs are widely used for well-to-well and depth correlation because natural radioactivity varies with lithology, especially shale/clay content.
- Geosteering uses real-time geological and geophysical logs to update borehole placement relative to geological targets; gamma ray is one common measurement in that workflow.
- Dynamic time warping is a general sequence-alignment method that can handle nonlinear shifts between similar signals, making it conceptually relevant for aligning horizontal `GR` signatures with vertical typewell `GR`.
- Geosteering research also frames the problem as fast, uncertain inversion from log responses to stratigraphic position; deterministic regression alone can miss multimodal possibilities.

## Project Direction

Immediate direction:

1. Promote the flat tail baseline as the first Kaggle-ready submission.
2. Keep GR/typewell correlation as an experimental second pipeline.
3. Add validation that uses the original `TVT_input` missing tail, not an arbitrary synthetic mask.
4. Build a guarded blend: use correlation only when its local match confidence is strong.
5. Treat tabular GBM as a residual/correction model, not the primary model.
