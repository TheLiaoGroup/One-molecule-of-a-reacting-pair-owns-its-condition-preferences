# Figure v3 display selection rules

Written before reading the selection results on 2026-10-21 under
the author-approved figure brief of 2026-10-21. These rules select illustrations only.
They do not change any analysis cohort, estimator or printed conclusion.

## Shared definitions

Use the current frozen analysis measurements. A product-forming pair has
maximum measured yield >= 0.10, as in the existing descriptive figure code.
Remove the intercept and response to the array's condition-mean profile
with the existing normal-equation OLS calculation. A preference barcode is
that residual profile in yield percentage points. Correlations use jointly
finite conditions; means of pairwise correlations are unweighted.
Keep the existing condition-composition order for all 95-column barcodes.
Never sort conditions, partners or arrays by a displayed result unless the
rule below explicitly defines that selection.

## Rule 1: second barcode of Figure 2B

Fix N#Cc1ccnc(Cl)c1 (2-chloro-4-cyanopyridine). Among its six non-shared
amine partners in the existing seven-partner illustration, choose the
amine whose preference correlation to the shared pair with
Cc1cc(C)nc(N)c1 (2-amino-4,6-dimethylpyridine) is closest to 0.27.
Resolve an exact tie by more measured conditions with yield > 0, then by
ascending original SMILES. Rank with unrounded correlations.

## Rule 2: representative owner molecule of Figure 3A

Use the existing frozen owner-role labels for all fifteen arrays. In each
array, first require at least four product-forming partners per candidate
owner molecule. For every eligible molecule, calculate the mean pairwise
preference correlation between its product-forming partners, using the
same descriptive definition as Figure 2. Select the molecule closest to
the median consistency of the eligible molecules in that owner role.
Resolve an exact tie by more product-forming partners, then by ascending
original SMILES. Rank with unrounded values.

If no molecule has four partners, use three; if none has three, use two.
Calculate the role median within the eligible set at the chosen threshold.
Record the threshold used separately for every array. If no molecule has
two partners or no finite consistency is available, stop that selection
and report it rather than invent another rule. This display choice does
not recompute an owner label or alter role-share estimates.

## Rule 3: array-level thin lines of Figure 5C

Use the existing 310 common held-out targets at every history depth.
Show array-level thin curves only for arrays with at least twenty of
these common targets. Include all arrays in the pooled thick curves.
Calculate curves from the existing individual SST and SSE records at
depths 1, 2, 3, 4 and 6, by 100 * (1 - sum(SSE) / sum(SST)).
Use no new fits or bootstrap draws. Retain the amplitude-inclusive
endpoint definition, distinct from the amplitude-removed Figure 5B.

## Non-selection orders

Figure 2A orders partners by existing display name (case-insensitive),
then original SMILES, without using r. Figure 4A orders the eight
structures by their existing amide-coupling T, as required by the
storyboard. These are presentation orders, not new result selections.
