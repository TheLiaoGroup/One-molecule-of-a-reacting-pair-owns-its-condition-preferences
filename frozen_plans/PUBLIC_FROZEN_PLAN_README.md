# Frozen analysis plans released with this article

These plans support the results reported in the article. Each listed folder keeps its frozen files and their SHA-256 records unchanged; only the files listed are released.

| Folder | Files | Supports |
|---|---|---|
| `reserve_validation/` | `PREREGISTRATION.md`, `PREREGISTRATION.sha256`, `reserve_validation.py` | Set A, the first out-of-array evaluation set: record-transfer test and ownership estimate |
| `cross_library/` | `PREREGISTRATION.md`, `PREREGISTRATION.sha256`, `cross_library_validation.py`, `OUTCOME.md` | Set B, the second out-of-array evaluation set: record-transfer test |
| `matched_control/` | `ANALYSIS_PLAN.md`, `design_and_input_lock_PUBLIC.json` | substituted histories |
| `fixed_condition_gap_15/` | `RULE.md`, `RULE.sha256` | fixed-condition gap over the fifteen arrays |
| `structure_in_place_of_record_15/` | `RULE.md`, `RULE.sha256` | structure-based recommender and similarity–preference association over the fifteen arrays |
| `directional_freeze_excerpt/` | `DIRECTIONAL_FREEZE_PUBLIC_EXCERPT.md`, `DIRECTIONAL_FREEZE_PUBLIC_EXCERPT.sha256` | same-molecule analysis: cohorts and criteria of Fig. 2, analysis rules of Figs 3c and 4, and the freeze time |

Beside the folders, two files accompany the plans of the out-of-array evaluation sets:

| File | What it is |
|---|---|
| `OUT_OF_ARRAY_EVALUATION_PROVENANCE_CORRECTION.md` | the correction that accompanies the plans of Set A and Set B and the Set B outcome record: what holds, the documented earlier outcome access, the post hoc Set B sensitivity and the historical statements it corrects |
| `SET_B_DOCUMENTED_PRIOR_SCORING.csv` | one row per Set B pair: documented prior pair-level scoring (124 pairs) and prior single-role losses (84) |

`matched_control/design_and_input_lock_PUBLIC.json` restates the hashes of that plan's frozen input lock, whose entries name files by paths on the computer used for the analysis; the frozen lock, with its own SHA-256 given in the record, is available to editors and reviewers on request.

Frozen plans are preserved as historical analysis records and may mention datasets or endpoints outside the finalized publication scope. Such references document analysis provenance and do not imply that the corresponding measurements or results are used or released in this article.

The plans of Set A and Set B and the Set B outcome record say that their pairs had not been opened or analysed before the freeze; those statements are historical and are corrected in `OUT_OF_ARRAY_EVALUATION_PROVENANCE_CORRECTION.md`. The verbatim excerpt of the same-molecule plan keeps that plan's wording about its future cohorts and about outcomes not read at its freeze; that wording is historical and does not state that no outcome of those cohorts was read elsewhere in the project lineage.
