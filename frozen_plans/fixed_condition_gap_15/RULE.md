# Fixed-condition gap over the 15 recurring-building-block arrays: frozen rule

Frozen 2026-10-04, before any 15-array value was computed or read. Decided by the author as part of the release
minimization: the three process-development C–N arrays should not become public data only to support one
Introduction number. The rule below is fixed before the single run. Whatever value it gives is reported and
accepted; no other estimand, array set or summary will be tried. Whether the sentence stays in the Introduction
is an editorial decision taken afterwards; it does not reopen this rule.

## Estimand (unchanged)

The per-array quantity is `gap_pp` from `opportunity(M)` in `06_CODE/Core/00_common/ctb/cost.py`, exactly as in
the published 18-array value:

1. Take the array's own pair × condition yield matrix `M`.
2. The best-on-average condition is the column with the highest mean yield.
3. Over the pairs measured under that condition, `gap_pp = 100 × mean(pair's own best yield − pair's yield under that condition)`.

The per-array value depends only on that array's matrix.

## Population

- Input data: the analysis training measurements, loaded by `load_analysis()`.
- Array matrices: built by `arrays()` from `06_CODE/Core/00_common/ctb/common.py`.
- Array set: the 15 arrays that remain after removing `UNIQUE_PAIR_ARRAYS` (the three process-development C–N arrays). The set must equal the 15 arrays of `05_DATA/Results/display_inputs/Anatomy_arrays.csv`.

## Summary

- Median of the 15 per-array `gap_pp` values, reported to one decimal in percentage points, with minimum and maximum.
- Single run.
- Each per-array value must equal the frozen `05_DATA/Results/primary/cost_arrays.csv` value to within 1e-9. A mismatch stops the run; it does not adjust anything.

## Output

`05_DATA/Results/directional/fixed_condition_gap_15.json`, written by
`06_CODE/Directional_Ownership/fixed_condition_gap_15.py`, which checks the SHA-256 of this file before computing.
