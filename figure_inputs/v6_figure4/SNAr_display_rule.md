# v6 draft Figures 4 and 5: display rules frozen before extraction

Approved storyboard: `../../../../04_Figure_Redesign/STORYBOARD_P2_v6_NatCatal.md`.
This draft does not change an analysis cohort, estimator, role label, or
existing selected molecule.

## Figure 4

- Eight amines: use all eight rows, in the existing frozen order
  M4, M8, M1, M2, M3, M5, M7, M6, from `molecule_context.xlsx`.
- Use the existing T values (variance shares), not Pearson agreement.
- Keep all 23 molecule-context comparisons and their existing plot order.
- SNAr: retain the exact existing illustrated pair from
  `06_CODE/Figures/array_shares_data.py`: halide `Cc1cc(F)ccc1C#N` and
  N-H nucleophile `c1cc2[nH]ccc2cn1`.
- Reciprocal-display extension, authorized for v6: the existing illustrated
  N-H nucleophile has only one product-forming partner and cannot supply a
  replacement experiment. Keep the existing illustrated halide unchanged.
  For the newly commissioned reciprocal block choose the N-H molecule with
  the largest number of product-forming partners; resolve an exact tie by
  ascending original SMILES. This selection uses no correlation or visual
  similarity. Include every product-forming partner (maximum measured
  yield at least 0.10), in ascending original partner SMILES. The two focal
  molecules are independent experiments; do not imply a shared intersection.
- Compute preferences with the existing `array_patterns` helper:
  product-forming pairs, intercept and slope on the screen mean removed,
  then yield percentage points. Use one shared condition order for both
  experiments, ordered by base-or-acid label, solvent label, condition ID.
- Preserve each exact source pair, condition ID and input hash in the
  draft source tables. The 45.6%/0.3% values remain the frozen screen-level
  role shares; they are not the examples' correlation coefficients.
- Move the existing 47-combination base/solvent inventory to a draft SI
  figure, retaining every original dot and source row.

## Figure 5

- Use the existing retained `transfer.xlsx` tables without fitting.
- Show every Set A and Set B screen estimate and both pooled estimates.
- Set A has 309 pairs and Set B has 535. The 167 versus 121 history-depth
  comparison is explicitly Set A only.
- Equal depth: use the existing exactly-two-partner estimates. Existing
  displayed differences and 95% intervals are 10.0 [5.1,13.1],
  7.3 [6.1,9.5], and 11.6 [10.6,12.5] percentage points. Preserve the
  already approved 7.3 display; the underlying unrounded estimate is
  retained in source data.
- Growth: use the same 310 common targets at all depths; preserve frozen
  Rule 3 (at least 20 common targets) for the two thin screen curves.
- Panels a/b concern residual condition preference; c concerns yield
  profiles including amplitude. State that distinction on the figure.

## Shared presentation

Native ChemDraw masters remain at 14.4 pt bond / 10 pt atom-label size.
Apply one common 0.7 print factor to every molecule: 7 pt main labels,
5.25 pt native subscripts, 10.08 pt printed bonds. No molecule-specific
scale, rotation, stretching or chemistry regeneration is allowed.
Ordinary text is 7 pt; compact auxiliary labels may be 5-6 pt; panel
letters are upright bold lowercase 8 pt. Fonts are read back from PDFs.
