# Cross-coupling conditions follow the halide, not the amine: data and code

This package holds the measurements analysed in the article, the files that fix how they were analysed (frozen analysis plans,
and the folds, draws and groups the analyses used), the code that produced the figures, and the figures and Source Data
themselves. The figures, the supplementary figures (Extended Data Figs. 1 to 10) and their Source Data can be rebuilt from this package alone.

## Contents

| Folder or file | What it holds |
|---|---|
| `data/` | the 244,179 released yield measurements in four files, each with its measurement status; the wells of the released pairs without a yield, with theirs; the condition table; the reagent table |
| `metadata/` | the outcome-free manifest of Set A and its release record; the frozen same-molecule cohorts (`DESIGN_COUNTS.csv`); the publication use of every released pair; frozen history folds and draws in `figure_display/` |
| `provenance/` | record-level matches to the published amide-coupling collection of Zhang et al. under the versioned matching rule PROV-1+A2; the rule, the counts and the matching code |
| `frozen_plans/` | the frozen analysis plans behind the reported results, with their SHA-256 records (`frozen_plans/PUBLIC_FROZEN_PLAN_README.md`), the definitions of the two evaluation sets and their per-pair Set B flags |
| `derived/` | the analysis tables made from `data/` (task construction), with `DERIVED_FILES.json` |
| `figure_inputs/` | three structures per depicted reaction, with the source screen and pair; products checked by a C-X/N-H substitution using the released reactants; `selection_v3/RULE.md`, the unchanged rules for the three new illustrations; retained computed inputs in `v5_reference_tables/` |
| `results/` | the retained results the figures are checked against, with `RESULTS_MANIFEST.csv`; `TEXT_CLAIMS.csv`, the statements of the text recomputed by the rebuild |
| `source_data/` | the Source Data of Figures 1-4 and Extended Data Figs. 1 to 10, with the released supplementary folds, draws and groups |
| `figures/` | Figures 1-4 and Extended Data Figs. 1 to 10 (PDF, SVG, main PNGs at 600 dpi); `structures/`, their ChemDraw structures and inventories |
| `code/` | the analysis and figure code; `code/release/`, the code of the two entry points below |
| `environment/` | the Python and package versions |
| `rebuild_all.py`, `verify_release.py` | the two entry points |
| `DATA_MANIFEST.csv`, `CHECKSUMS.sha256` | every file of the package with its size, SHA-256 and role |
| `LICENSE_DATA_CC-BY-4.0.txt`, `LICENSE_CODE_MIT.txt` | the licences: CC BY 4.0 for the data and documents, MIT for the code |
| `.gitattributes` | Git settings: no line-ending conversion, so a clone holds the bytes listed in `CHECKSUMS.sha256` |

## The measurements: 244,179 numeric yields in four files

| Unit | File | Records | Pairs | What they are |
|---|---|---|---|---|
| A1 | `data/measurements_A1_analysis_arrays.csv.gz` | 177,039 | 2,207 | A1 holds the measurements of the 15 analysis screens (the second copy of the duplicated Hiyama condition is in A3) |
| A3 | `data/measurements_A3_hiyama_duplicate.csv.gz` | 74 | 74 | Hiyama condition C46, measured as a duplicate of C43; used for repeatability and noise calibration |
| SA1 | `data/measurements_SA1_set_A.csv.gz` | 36,499 | 420 | the pairs of Set A, the first out-of-screen evaluation set, that enter the record-transfer test (309), the three-condition decision (396) or the ownership re-estimate (217 product-forming amide pairs) |
| SB1 | `data/measurements_SB1_set_B.csv.gz` | 30,567 | 535 | every pair of Set B, the second out-of-screen evaluation set |
| | | **244,179** | | |

Each record has a `public_record_id` of the form `P2-<unit>-<seven-digit number>`, numbered in file order and unique across
the four files. The Hiyama duplicate is a separate measurement and keeps its own identifier; records with the same chemistry are never
merged. The other columns are the unit, the screen (`screen`), the reaction class, the plate (screen measurements only), the
condition identifier, the substrate pair and its two substrates as SMILES, the yield as recorded (a fraction between 0 and
1), its `measurement_status` and the inherited flag `check_passed` (90 records carry `False`; the flag imposes no exclusion).
The status is `quantified`, or `zero_by_rule` for a well whose screen detected no product and which therefore has yield 0;
a well whose product was detected but not quantified, or which has no screen record, is missing, never zero.

`data/wells_without_yield.csv` lists the wells of the released pairs that have no yield, with their unit and status
(`detected_not_quantified` or `no_screen_record`); 378 of them lie in the fifteen screens. They are not measurements: they
have no identifier and are not among the 244,179.

`data/conditions.csv` gives the 1,192 conditions named by the records with their full compositions (reagents with CAS
number, equivalents, name and identifier; solvent; temperature; time); a condition is identified by `rxn_class` and
`condition_id`. `data/reagents.csv` gives the 401 reagent entries referenced in the reagent slots of those conditions;
solvents are named in the condition table itself.

For Set A, `metadata/set_A_frozen_manifest.csv` lists all 442 frozen pairs without outcomes, and
`metadata/set_A_release.json` gives the SHA-256 of the frozen data file, the rule that selects the released pairs and the
SHA-256 of the released subset.

Set A and Set B are the article's two out-of-screen evaluation sets: pairs outside the fifteen analysis screens, excluded from
model fitting and scored under frozen complete-set plans. `frozen_plans/OUT_OF_ARRAY_EVALUATION_PROVENANCE_CORRECTION.md`
describes earlier outcome access to parts of both sets;
`frozen_plans/SET_B_DOCUMENTED_PRIOR_SCORING.csv` flags the 124 Set B pairs concerned.

## Provenance of the records

`provenance/P2_RECORD_PROVENANCE.csv` has one row per released measurement, in the order of the four files (unit A1 first),
so its row for record *i* of a unit is `P2-<unit>-<i>`. For each record it says whether the record matches, under the
versioned rule PROV-1+A2 (`provenance/PROV-1+A2_PUBLIC_RULE.md`), a record of the previously published amide-coupling
collection of Zhang et al.; the totals are in `provenance/P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json`.
The columns are `unit`, `screen`, `condition_id`, `sub_1_smiles`, `sub_2_smiles`, `yield`, `matches_zhang2025`
and `match_rule_version`.

Parts of these measurements also appear in other public releases of the same corpus by our group.

A match is an exact record match after harmonising the reaction library, the molecular identities, the condition identity and
the yield representation. It does not establish that two records come from the same physical experiment. No first disclosure is
claimed for any released measurement.

The Zhang et al. collection is not redistributed here. Its repository, version, file path and checksum are recorded in
the rule and the counts. To re-derive the flags, supply its reference file:

    python rebuild_all.py --zhang <all_HTE_with_condition.csv of the recorded version>

Without the reference file, the rebuild checks record alignment, the Zhang flags and their sum against the released counts,
and rebuilds the figures all the same.

## Verify only, or rebuild

Install the pinned Python packages:

    python -m pip install -r environment/requirements.txt

All four main figures use matplotlib with the released ChemDraw artwork. The reference environment for exact byte
comparisons is Windows x64 with the listed Python packages and Arial files. Exact released bytes are checked in this
environment; byte identity on other operating systems has not been established.

    python verify_release.py

checks the package as it stands, without rebuilding or writing anything: the hashes of every file, the 244,179 records,
their identifiers and statuses, the wells without a yield, the condition and reagent tables, the release rule of Set A, the
alignment of the provenance flags, the derived tables, the results manifest, the publication use of every measurement, the
statements of the text and the frozen-plan records.

    python rebuild_all.py

rebuilds everything from this package (about 15 minutes with 6 processes on the computer used for the study; `--jobs` sets the
number). It makes the derived
tables again from `data/`, runs the four main and ten supplementary figure generators, the re-fit of Extended Data Fig. 10, the owner labels fixed without the scored pairs
(`code/Checks/nested_owner.py`) the owner margins, equal-role null and SNAr class proxy (`code/Checks/role_null_checks.py`) and the per-screen tables of the
Supplementary Information (`code/Checks/si_tables_checks.py`), recomputes the
statements of the text and the publication use of every measurement, and compares every rebuilt figure, Source Data file and
provenance record with the released one. The four main-figure audits independently recalculate the displayed values and
check the actual series of the rebuilt figures (`code/Checks/audit_figure1.py` to `audit_figure4.py`, `--values-only`).
Every Python process of the rebuild runs under a file-access guard: it may read this
package, Python and the system fonts, and write only in `_rebuild/`. Only this Python executable may start child
processes, and Python network connections are refused. Python file openings are logged. The
outcome is written to `_rebuild/REBUILD_REPORT.json`; `_rebuild/` is not part of the release.

On the reference environment (`environment/`) every rebuilt figure and Source Data file is byte-identical to the released
one. A provenance record (`*_SourceData_provenance.json`) is identical apart from its list of inputs, which names the files
the rebuild read: the derived tables and filtered results of this package where the original build read internal files, and
the released folds and draws where it read the internal fold registry. `rebuild_all.py` accepts exactly these differences
and no other.

## How the code finds its inputs

The code under `code/` is the code that produced the figures, unchanged, and it addresses its inputs by the paths of the
original working layout. `rebuild_all.py` therefore builds a work tree in `_rebuild/tree/` from copies of the package files
at those paths (`code/release/package_lib.py`, `stage_map`), and runs the generators in public mode
(`P2_RELEASE_MODE=public`), in which the hidden state (the history-depth folds and draws, supplementary folds and draws,
and the molecule groups, condition folds and pair halves of Extended Data Fig. 10) is read from
the released metadata and Source Data instead of the internal fold registry. The shared modules also hold functions for analyses that are
not part of the article; the rebuild does not run them.

The condition metadata of `data/conditions.csv` is also copied to its original figure-input path in the work tree;
the main-figure layout and label helpers and the retained scientific helpers are included under `code/Figures/`.
`figure_inputs/v5_reference_tables/` contains five retained computed workbooks (`overview.xlsx`, `preference.xlsx`,
`screen_shares.xlsx`, `molecule_context.xlsx`, and `transfer.xlsx`) and `preference_payload.json`, with their source hashes
in `MANIFEST.json`. These are immutable
analysis inputs, copied to `05_DATA/Analysis/figure_display/v5_reference_tables/` in the work tree. They are separate from
the Source Data generated by the four current figures. Figure 1 recalculates its preference payload from the measurements
and compares it with the retained reference payload. Rebuild step 14 independently checks the displayed scientific values.
Extended Data display numbers follow first citation in the article: Extended Data Fig. 3 uses the condition-halves generator `ed04`
and stable `FigS4` figure/Source Data files; Extended Data Fig. 4 uses the leave-one-molecule-out generator `ed03` and stable `FigS3`
files. The remaining Extended Data display numbers match their stable file numbers.
The Figure 4 generator also regenerates `SetA_whole_array_bootstrap_draws.csv` and
`SetA_owner_block_bootstrap_draws.csv` under `source_data/Figure4/`. The history folds and history draws are frozen inputs,
copied without changing their bytes from `metadata/figure_display/` to `05_DATA/Analysis/figure_display/` in the work tree.
They are not outputs regenerated by Figure 4. The Figure 4 workbook retains the established `Fig4a` and `Fig4b` sheet
identities for the nested-owner comparison; these sheet names identify analyses rather than current panel numbers.

## Licences

The source code is released under the MIT licence (`LICENSE_CODE_MIT.txt`, Copyright (c) 2026 Kuangbiao Liao): `code/`,
`rebuild_all.py`, `verify_release.py`, `provenance/provenance_overlap_public.py` and the frozen scoring code in
`frozen_plans/` (`reserve_validation/reserve_validation.py`, `cross_library/cross_library_validation.py`). Everything else (the
measurements, tables, metadata, provenance records, analysis plans and evaluation-set definitions, derived tables, results, Source Data,
figures, structures and documentation) is released under the Creative Commons Attribution 4.0 International licence
(`LICENSE_DATA_CC-BY-4.0.txt`). Neither licence covers the amide-coupling collection of Zhang et al., which is not redistributed here.

## Fonts

No font file is included: Arial is not redistributed. `environment/fonts.json` records the versions and SHA-256 of the Arial
files the released figures were drawn with (regular, Version 6.90; bold and italic, Version 7.00). The numeric results and the
Source Data do not depend on fonts; byte-identical figure files (PDF, SVG, PNG) require the recorded Arial files.

## Additional measurement and analysis details

The full collection contains 410,983 records in 20 reaction tables. Removing 396 unreconciled records leaves 410,587; excluding 3,678 halide-exchange records leaves 406,909 records in 19 tables. The 15 analysis screens come from 14 tables and cover 14 transformations.

The 244,179 numeric yields comprise 177,039 analysis-screen measurements, 74 Hiyama duplicates, 36,499 measurements from 420 Set A pairs and 30,567 from 535 Set B pairs. The data files preserve all four measurement statuses, the 1,192 conditions, 401 reagent entries and 442-pair Set A manifest.

For the full history-depth records, 496 owner records from 63 owner histories give endpoints 8.6% and 23.6%; 751 partner records from 87 partner histories give 0.6% and 4.1% (Extended Data Fig. 7; Supplementary Note 11). The weighted T recombination covers all 30 role shares, with largest difference 1.4 × 10⁻¹⁶; these implementation checks are in the code.
