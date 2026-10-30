# Structure in place of a record over the 15 recurring-building-block arrays: frozen rule

Frozen 2026-10-04, before any 15-array value of this item was computed or read.

## Origin

The author approved the rule for the four exposure regimes. Its purpose: the Extended Data item and its Discussion
sentence should rest only on the 15 arrays whose data will be released, so that the three process-development
C–N arrays (13,866 measurements) can stay internal. Section B adds the item's one other number, the
similarity–pattern association, under the same population restriction. Without it the item would still carry a
value pooled over arrays that include two process-development arrays.

## Common terms

- **Inputs.** Frozen outputs only, checked by SHA-256 before use:

  | File | SHA-256 |
  |---|---|
  | `05_DATA/Results/primary/fig3_regime_cases.csv` (per case) | 10f9bd7fd6f6ebb74fc7a6263d64c1dc41ef838b7da440839777021ebe8e000a |
  | `05_DATA/Results/primary/fig3_generalisation_regimes.csv` (per array) | 007d8e978920641dcede3909aa796a62906ba09ad414db1f87f62f224c758c9b |
  | `05_DATA/Results/primary/structure_association.csv` (per array) | 47e752a8a36330062c3d18bc88ea7b6076b5707b0c126b32cbdc362035f3bb6c |

- **Population.** Drop every row whose array is one of `UNIQUE_PAIR_ARRAYS` in `06_CODE/Core/00_common/ctb/common.py`, the three process-development C–N arrays. Keep all other rows.
- **No re-fitting.** No model is re-fitted, no fold or partition is rebuilt, no hyperparameter is chosen again, and no target eligibility changes.
- **Stop condition.** If any statistic below cannot be computed from the input rows alone, the run stops and nothing is reported.
- **Identity check before the restriction.** The same code applied to all rows must reproduce the published 18-array values to within 1e-12:
  - `05_DATA/Results/display_inputs/Figure4d_structure_regimes.csv` (b42e5f89…);
  - the association block of `Structure_regimes.json` (2658aa7e…).

  A mismatch stops the run.
- **Single run.** Every value is accepted whatever its direction or size. The 15-array values replace the 18-array values completely; the old values are not kept beside them and are not restored if the new ones are weak.

## A. The four exposure regimes (approved by the author)

Unchanged:

- the exposure-regime definitions: warm/warm, new/known, known/new, new/new;
- the fixed comparator;
- the scoring rule;
- the case weights;
- the aggregation estimator;
- the resampling unit.

For each regime, with target `p_oof_centred` and the scored cases (`scorable`), exactly as `summarise()` in `06_CODE/Core/00_common/ctb/regimes.py`:

- **Gains.** The pooled mean gain and 95% interval for `gain_greedy` (against the complementary trio, the reported quantity) and for `gain_mean`. Both come from `whole_array_interval()`: pooled mean = sum / count over cases, whole arrays resampled, seed 20260920, 10,000 draws.
- **Counts.** The number of arrays with records; scored unique pairs summed per array; scored records.
- **R².** The median over arrays of `r2_amplitude` and `r2_p_oof_centred`.

## B. The similarity–pattern association (added to the approved rule)

- **Arrays.** The rows of `structure_association.csv` that remain after the population restriction: 14 arrays (Hiyama has no row).
- **Statistics.** For `rho`, `rho_preference` and `rho_preference_c`, exactly the fields that `structure.py` reports:
  - median ρ;
  - the number of arrays with positive ρ;
  - the number with q < 0.05, where q is recomputed with `bh()` of `common.py` over these arrays' p values, since the correction set changes;
  - median ρ without shared molecules, and the number of arrays where it is positive.

## Output

`05_DATA/Results/directional/structure_in_place_of_record_15.json`, written by
`06_CODE/Directional_Ownership/structure_in_place_of_record_15.py`, which checks this file's SHA-256 and the input
hashes before computing.
