# Out-of-array evaluation sets: provenance correction

This correction accompanies the historical frozen plans of the two out-of-array evaluation sets of the article, `reserve_validation/` (Set A) and `cross_library/` (Set B), and the Set B outcome record. Those files are released unchanged, byte for byte, with their SHA-256 records; this file corrects what they say about earlier access to the outcomes. Times are UTC; the dates written in the plans are local dates (UTC−4).

## What holds

- Neither set contributed any row to the analysis-array fits that learn the owner and partner histories. No (array, pair) of either set occurs in the analysed arrays, and the data reader refuses their rows for fitting.
- For each set, the complete-set endpoint, estimator, resampling scheme and decision rule were frozen, with the SHA-256 of the plan, the scoring code, the evaluation data and the outcome-free target manifest, before the reported complete-set endpoint was computed.

| | Set A | Set B |
|---|---|---|
| Pairs | 442 frozen; 309 record-transfer targets | 535 |
| Plan, code, data and target manifest hashed | 2026-10-01T15:00:38Z | 2026-10-02T01:35:12Z |
| Complete-set endpoint first computed | 2026-10-01T15:00:43Z | 2026-10-02T01:35:18Z (results written 01:35:22Z) |
| Earlier outcome access, first documented | 2026-09-26T01:55:07Z | 2026-09-28T16:26:51Z |
| Pairs with pair-level outcome-derived quantities before the freeze | 296 of 442, among them 223 of the 309 targets | 124 of 535 (4,912 measurements) |

## Documented earlier outcome access

A provenance audit made after the evaluations found that outcomes from portions of both sets had been accessed in earlier analyses in the direct project lineage of the article. The two sets are therefore out-of-array corroboration rather than outcome-blind validation.

**Set A.** On 2026-09-26 a predecessor analysis read the yields of every pair of the platform's reserve partition, from which Set A was later drawn, to screen eligibility. It derived pair-level quantities from the outcomes of 296 of the 442 Set A pairs, among them 223 of the 309 current record-transfer targets: a quantity computed from the pair's yields for 295 pairs and, for 237, the yields reached by three conditions ranked from the owner's record. Because the screen read the yields of every pair, and which pairs then received pair-level quantities depended on those yields, no outcome-unexposed subset of Set A is defined and none is analysed.

**Set B.** On 2026-09-28 an earlier evaluation in the same lineage read 4,912 measurements of 124 of the 535 Set B pairs and computed pair-level losses for them; on 2026-10-01 single-role losses were computed for 84 of these pairs. `SET_B_DOCUMENTED_PRIOR_SCORING.csv` flags them (one row per Set B pair: array, pair, documented prior pair-level scoring, documented prior single-role losses). Before the Set B freeze, array-level mean yields were also computed over populations that contained every Set B row, as the plan's list of quantities seen before freezing records; they are aggregates, not pair-level quantities.

## Post hoc provenance sensitivity (Set B)

Post hoc; not a preregistered validation. The frozen Set B estimator (seed 20261007, 2,000 draws per resampling scheme), applied unchanged to the full set and to its two parts. The 411 pairs are those without documented prior pair-level scoring exposure in the audited project lineage.

| Set B subset | Pairs | D | Owner-cluster interval | Whole-array interval | Arrays > 0 |
|---|---|---|---|---|---|
| Full | 535 | 0.089 | 0.057 to 0.119 | 0.063 to 0.111 | 11/13 |
| Excluding documented prior pair-level scoring | 411 | 0.098 | 0.061 to 0.134 | 0.079 to 0.110 | 11/13 |
| Documented prior pair-level scoring | 124 | 0.039 | −0.026 to 0.098 | −0.002 to 0.151 | 8/12 |

Arrays > 0 counts the arrays with a defined D; for the SNAr pairs of the last row the pattern has no variance, so D is undefined there.

## Statements in the historical files that do not hold

| File | Lines | Statement | Correction |
|---|---|---|---|
| `reserve_validation/PREREGISTRATION.md` | 1 | “pairs this paper has never analysed” | Does not hold in the direct project lineage: a predecessor analysis read the yields of every reserve-partition pair on 2026-09-26 and derived pair-level quantities from the outcomes of 296 of the 442 Set A pairs. |
| `reserve_validation/PREREGISTRATION.md` | 11 | “hold on substrate pairs that entered no analysis in this paper” | Does not hold in the direct project lineage: a predecessor analysis read the yields of every reserve-partition pair on 2026-09-26 and derived pair-level quantities from the outcomes of 296 of the 442 Set A pairs. |
| `reserve_validation/PREREGISTRATION.md` | 43 | “No single-role profile, no share, no recommendation and no endpoint has been computed on these rows.” | Does not hold for 237 pairs, for which three conditions ranked from the owner's record had been scored by their yields; pair-level outcome-derived quantities existed for 296 pairs. The plan's own complete-set endpoints were first computed after its freeze. |
| `reserve_validation/PREREGISTRATION.md` | 125–126 | “they were never opened by this paper” | Does not hold in the direct project lineage: a predecessor analysis read the yields of every reserve-partition pair on 2026-09-26 and derived pair-level quantities from the outcomes of 296 of the 442 Set A pairs. |
| `cross_library/PREREGISTRATION.md` | 19–20 | “have entered no analysis in this work” | Does not hold for 124 of the 535 pairs (4,912 measurements), whose yields were read and scored on 2026-09-28. |
| `cross_library/PREREGISTRATION.md` | 35–36 | “No single-role profile, no loss and no endpoint has been computed on them.” | Does not hold for 124 pairs: joint-profile losses had been computed for all 124 and three-condition yield comparisons for 19 on 2026-09-28, and single-role losses for 84 on 2026-10-01. The plan's own complete-set endpoint was first computed after its freeze. |
| `cross_library/PREREGISTRATION.md` | 90 | “were never opened by this paper” | Does not hold for 124 of the 535 pairs (4,912 measurements), whose yields were read and scored on 2026-09-28. |
| `cross_library/OUTCOME.md` | 6 | “none of them part of any analysis in this work” | Does not hold for 124 of the 535 pairs (4,912 measurements), whose yields were read and scored on 2026-09-28. |
| `cross_library/OUTCOME.md` | 48 | “The two arrays that do not favour the owner are the two lowest-yielding populations in the sealed set.” | Does not hold: by the Set B rows SNAr is the lowest-yielding array and Chan–Lam the fourth lowest, after Hiyama and reductive amination (table below). |
| `cross_library/OUTCOME.md` | 58–59 | “844 substrate pairs that entered no analysis in this work were scored under rules frozen before the measurements were opened” | Does not hold as written: outcomes of 296 Set A pairs and 124 Set B pairs had been accessed earlier in the direct project lineage. What holds is that both sets were excluded from model fitting and that each set's complete-set endpoint procedure was frozen before that endpoint was computed. |

## Mean yields in the Set B outcome record

Nine of the thirteen values in the mean-yield column of `cross_library/OUTCOME.md` (line 32) are means of each array's condition library over a larger population, computed before the freeze, not means over the Set B rows. The D column is the frozen run's.

| Array | D | Mean yield in OUTCOME.md | Mean yield over the Set B rows |
|---|---|---|---|
| Reductive amination | +0.239 | 0.047 | 0.047 |
| Esterification | +0.217 | 0.170 | 0.177 |
| Sonogashira | +0.189 | 0.278 | 0.335 |
| C–N (Pd ligand) | +0.121 | 0.159 | 0.217 |
| Heck | +0.116 | 0.210 | 0.239 |
| Amide coupling | +0.099 | 0.285 | 0.312 |
| Hiyama | +0.062 | 0.035 | 0.035 |
| Kumada | +0.060 | 0.213 | 0.248 |
| Suzuki | +0.054 | 0.349 | 0.356 |
| Grignard | +0.051 | 0.342 | 0.310 |
| Ullmann C–O | +0.041 | 0.154 | 0.214 |
| Chan–Lam | −0.006 | 0.052 | 0.052 |
| SNAr | −0.022 | 0.018 | 0.018 |

Supplementary Note 10 reports the two arrays that do not favour the owner, Chan–Lam and SNAr, with their D and their mean yields over the Set B rows, without a ranking.

## Scope

The audit documents exposure in the audited direct project lineage and does not claim an exhaustive census of every antecedent research project. The detailed internal provenance record (the machine evidence behind each time and the per-pair records of both sets) is available to editors and reviewers on request.
