# Same-molecule analysis: frozen plan, public excerpt

This file reproduces Sections 1–7, the freeze marker, Amendment 1 and Notes 1–3 of the frozen analysis plan of the same-molecule analysis, verbatim except for the English translation of one historical manuscript path in Amendment 1, item 5. Section 8 concerns future cohorts not used in this article and is omitted from the public excerpt; Sections 9–10 (the run order, and the internal files read after the freeze with their paths and hashes) are omitted with it. The complete plan can be provided to editors and reviewers for research-integrity verification; it is not part of the public repository.

- SHA-256 of the complete plan: 9bfb2f20edbc9d4ff60d7fb0b365954267690bc4ac702a93b0023dcf7cf6460b
- SHA-256 of this excerpt: `DIRECTIONAL_FREEZE_PUBLIC_EXCERPT.sha256`
- The freeze marker, the amendment and the notes state SHA-256 values of the complete plan above them; those values refer to the complete plan. The marker's hash of `DESIGN_COUNTS.csv` is that of the released cohort file.

---

## 1. Question

Does the same molecule carry more of its condition preference across partners when it sits on the owner side of
one reaction than when it sits on the non-owner side of another? The result decides whether the paper is rebuilt
around "the reaction assigns the owner".

## 2. What was read before this freeze

Read: molecule and condition identities, cell masks, and one yield-derived quantity, the published
product-forming gate (pair maximum yield >= 0.10), reproduced exactly (1,542 pairs; every per-array count equals
`Anatomy_arrays.csv`). Already known from the published P2 work: role shares and owner labels, Fig. 1c pair
correlations, Set A/B results, the single-role R2 in `history_arrays.csv` (including the five arrays where
additive residuals favour the non-owner) and `amplitude_arrays.csv`.

Not read: any molecule-level quantity T(m, r); the role split of `04_RESULTS/primary/anatomy_depth.csv`; any
yield of the fresh cohorts in section 8. `code/design_freeze.py` reproduces every count below and asserts it.

## 3. Definitions (all on the published estimator, `anatomy.py`)

- **Data.** Analysis partition of `02_DATA/01_CANONICAL/measurements.csv.gz`; the 15 recurring-molecule arrays
  (the three Buchwald-Hartwig(CMC) arrays excluded); cells with missing yield and the duplicated Hiyama C46
  excluded. Within an array a molecule is its subpair spelling, as in `anatomy.py`; across arrays it is the RDKit
  2025.03.2 canonical isomeric SMILES. Each canonical molecule has one spelling within each array (asserted).
- **Pattern.** Within array r, on product-forming pairs only: s_c = mean yield of condition c; each pair's yields
  are regressed on s (intercept and slope, ordinary least squares, observed conditions); residuals e_ic.
- **Array variance.** V_r = mean over product-forming pairs i of (sum_c e_ic^2 / n_i).
- **Pair-pair product.** C_ij = mean of e_ic * e_jc over the conditions observed for both pairs.
- **Molecule-level transport.** For molecule m in role R of array r, with S = its product-forming pairs:
  T(m, r) = mean of C_ij over ordered pairs i != j in S that share at least one observed condition, divided by V_r.
  Defined when m has >= 2 product-forming partners in r.
- **Relation to the published shares.** The published share of role R is the average of T(m, r) over role-R
  molecules weighted by each molecule's number of ordered pair-pairs. The G1 statistics weight molecules equally:
  each identical molecule contributes once. They are a molecule-level estimand, not the published share.
- **Owner labels.** Published `owner_role` in `Anatomy_arrays.csv`. A clear-owner array has owner share >= twice
  the partner share (`owner_fraction` >= 2/3): 12 arrays; not clear: Chan-Lam|1, Hiyama|1, Reductive-amination|1.
  m is in an owner context in r when its role in r is the owner role; otherwise a non-owner context.

**Implementation check, run first.** For every array and role, the pair-pair-weighted average of T(m, r) must equal
the published share1/share2 to within 1e-9. If it fails, stop; the code is corrected against `anatomy.py` only.

## 4. Cohorts (frozen in `DESIGN_COUNTS.csv`)

| Set | Definition | n | SHA-256 of sorted members |
|---|---|---|---|
| G1 | >= 2 product-forming partners in at least one owner and one non-owner context, both in clear-owner arrays | 23 | 9147efe7e2897fa4347ca1460790bfd3828d768403d16800e225c596581c012c |
| amine stratum | G1 molecules whose owner context is amide coupling | 19 | |
| acid stratum | G1 molecules whose owner context is esterification | 4 | |
| triple context | >= 2 product-forming partners in amide coupling, BH\|1 and BH\|2 | 8 | cf83093b88f6d909a783e2fcb92753be1c6fd7f1987c3cd44e3ebed49190ffeb |
| G1, all contexts | as G1 but any array | 29 | cae0e25dd7ad20b95e5f33434dd761588beb1d72dfbce8ae9be2dea054e8774e |
| owner-owner control | owner in >= 2 clear arrays, never non-owner | 38 | bef47a945d4b8da3a4c3508bc35b57dc9ad7e260acd6084608f75d5f9087b568 |
| non-owner control | non-owner in >= 2 clear arrays, never owner | 15 | 27628aa3471e12124d3ae8ed95166c1d3850422d95bcff0050a4afddb0192da8 |

Amine non-owner contexts: BH|1 only 8; BH|1 and BH|2 8; BH|2 only 1; BH|1 and Sonogashira 1; SNAr 1. Acids: all four
have amide coupling as the non-owner context. Each molecule keeps its mechanistic role across its contexts (N-H
nucleophile; activated carboxylic acid), except one amine that enters Sonogashira coupling as the alkyne; only the
reaction changes. The triple-context molecules are
4-methoxyaniline, 4- and 3-(methylthio)aniline, 2,4-dimethylaniline, 4-methylaniline, 2-methylaniline,
5-methylpyridin-2-amine and quinoxalin-6-amine. BH|1 (95 conditions) and BH|2 (89) share no condition identifier.

## 5. Primary analysis and criteria

For each G1 molecule: T_O(m) = mean of T(m, r) over its clear owner contexts; T_N(m) = mean over its clear
non-owner contexts; D_m = T_O(m) - T_N(m). Each molecule gives one D_m.

- **A. Pooled.** Mean of D_m over the 23 molecules; 95% percentile interval from 10,000 draws resampling molecules
  (seed 20261004). Holds if the mean > 0 and the lower bound > 0.
- **B. Breadth.** Amine mean with its own interval (10,000 draws within the stratum); the four acid values and their
  mean, without an interval claim. Holds if the amine mean > 0 and the acid mean > 0.
  Reported with B, not a criterion: the reaction-pair-balanced mean 0.5 x amine mean + 0.5 x acid mean, 10,000
  stratified draws (resampling within each stratum).
- **C. Same molecule, new library versus new reaction.** For each triple-context amine:
  R_m = T(amide) - [T(BH|1) + T(BH|2)] / 2. Holds if mean R_m > 0 and R_m > 0 for at least 6 of 8 molecules.
  The library contrast T(BH|1) - T(BH|2) is reported with a molecule-bootstrap interval as a description only;
  eight molecules support no equivalence claim.
- **D. Self-inclusion.** For each G1 molecule and each context array it uses: delete every product-forming pair
  containing that molecule from the array, recompute s, residuals, V and both role shares with the published
  estimator, and record the owner (larger share; ties to role 1 as in `anatomy.py`). Holds if no label changes. If
  labels change, D still holds when criterion A, recomputed without the affected molecule-context entries
  (molecules left without an owner or a non-owner context dropped), still holds. The number of changes is
  reported either way.

Reported with the primary result, not criteria: median D_m; the fraction of D_m > 0; every D_m with its structure;
the 29-molecule all-context version (includes the four boronic acids, whose Chan-Lam owner label is not clear);
T in both contexts for the 38 and 15 control molecules; for owner-owner molecules, the Spearman correlation of T
between their two contexts with the most product-forming partners (ties by array name), which tests whether
strength within a role is partly molecular; and the Spearman correlation between D_m and the difference in mean
partner Tanimoto between owner and non-owner contexts (`DESIGN_COUNTS.csv`).

## 6. Decision

- **Full redesign:** A, B, C and D all hold.
- **Partial redesign:** A and D hold; B or C fails. The claim is limited to "reaction context can reassign ownership
  in matched amines"; Figures 1 and 2 may be rebuilt, Figure 3 stays on the existing Set A/B validation.
- **No full redesign:** A fails, or D fails. The current P2 architecture stays; it takes the reciprocal-replacement
  framing, the history-depth design table and the literature and figure improvements.

The analyses in section 7 and the fresh cohorts in section 8 cannot change this classification. No criterion,
cohort or estimator changes after the marker below.

## 7. Non-gating analyses

1. **Equal history depth for Figure 3 (post hoc; Set A/B outcomes already known).** The frozen Set A/B transfer
   estimator (`partner_external.train/score`: all training pairs, canonical identities, slope-adjusted pattern,
   shrinkage 4, >= 8 cells) with each role's profile built from exactly two of the molecule's training pairs: all
   two-pair subsets if there are at most 45, otherwise the 45 with the smallest SHA-256 of the sorted partner
   identities; target SSE averaged over subsets. Same 309 and 535 scored targets, 2,000 whole-array draws per set
   (seed 20261004). Labelled a post hoc matched-depth sensitivity analysis.
2. **Equal-depth transport in the arrays (Figure 4a candidate).** Targets: the 910 product-forming pairs in 12
   arrays whose two molecules each have >= 2 other product-forming partners (arrays with >= 10 such targets; SHA-256
   cc2426bc0740263cbfadb67806befb28815979af1bbf2a424112a84fdd2da03a; C-O 1, Hiyama 2 and Chan-Lam 7 targets fall
   below). For each target, s and residuals are recomputed without it; owner and partner profiles are sums of the
   residuals of two history pairs divided by (observed count + 4); target and profiles are projected off intercept
   and s over the target's observed conditions; subsets as in 7.1. Statistic: (sum SSE_partner - sum SSE_owner) /
   sum SST, pooled, with per-array signs; 10,000 whole-array draws (seed 20261005).
3. **Three-condition decision at equal depth (Figure 4b candidate).** Same targets and histories. Primary ranking:
   s_c + slope-adjusted single-role profile_c (owner or partner), top three, ties broken by condition identifier.
   Comparator: P2's complementary fixed trio fitted on all other pairs of the array. A target is scored only if
   every condition in the union of the three panels is observed for it. Outcome: best measured yield in each panel.
   Contrasts: owner minus partner (primary), owner minus trio, partner minus trio; subsets averaged within target,
   targets weighted equally; 10,000 whole-array draws (seed 20261006). Secondary: the same with additive-residual
   profiles, which also carry response amplitude.
4. **Owner-depth curve (descriptive).** Amide coupling, Suzuki and BH|1, the only arrays where >= 50 targets have an
   owner molecule with >= 6 other product-forming partners (350, 81, 66). K = 1, 2, 3, 4, 6; five K-pair subsets
   per target by smallest SHA-256; ranking as 7.3 against the trio; the same for partner molecules with >= 6
   partners (369, 212, 41). Pointwise 95% intervals from 10,000 draws resampling owner molecules within arrays. The
   whole curve is shown; the first K whose pointwise lower bound exceeds zero is stated as a description, not a
   threshold.
5. **`anatomy_depth.csv` by role (descriptive).** Pooled R2_k = 1 - sum SSE_k / sum SST over the owner-role and the
   partner-role records, k = 1, 2, 3, 4, 6, overall and per array where both roles have records. These records use
   additive residuals.
6. **Condition library (Extended Data).** BH|1 versus BH|2 as above. Condition halves: in each array, conditions
   ordered by SHA-256 of the condition identifier and assigned alternately; gate unchanged; pattern and shares
   recomputed within each half; owner per half compared with the published owner.
7. **Amplitude ownership (exploratory).** The slope of each product-forming pair from the pattern regression,
   centred within the array; the amplitude share of role R is the mean over ordered pair-pairs sharing a role-R
   molecule of the product of centred slopes, divided by their variance. Agreement with the preference owner is
   tabulated over the 15 arrays. Extended Data or Discussion only if a clear pattern holds across arrays.

*[Sections 8–10 of the complete plan are omitted here.]*

**FREEZE COMPLETE — NEW ENDPOINT OUTCOMES NOT READ**

- Time: 2026-10-04 13:10 -0400
- SHA-256 of this file above the separator line: 54d9da9d39e1eacad474db46e1a65e041bb78b13475827c9f99cd18a907b1f38
- SHA-256 of `DESIGN_COUNTS.csv`: 707ec422fd493e7a95653d08b57536f9b76e036ea3938509996e11e98b647187
- SHA-256 of `code/design_freeze.py`: e01cb0e1eeb2143f79f6909790b3f76301938efffa9ef014ecad1df5553078c6

An amendment before any section 5, 7 or 8 outcome is read is appended below this line with its time; none after.

## Amendment 1 — 2026-10-04 13:26 -0400, before any section 5, 7 or 8 outcome was read

SHA-256 of this file above this heading: f086ef00cd6f55d1ccc486b610015842dc298c827ceb792c7392b9a242c15869

1. **Criterion C** now requires all three: mean R_m > 0; R_m > 0 for at least 6 of the 8 molecules; and both
   mean(T_amide - T_BH1) > 0 and mean(T_amide - T_BH2) > 0 over the 8 molecules. The last two need no interval.
   No equivalence claim between BH|1 and BH|2 is made.
2. **Estimand, restated.** The published role share is a pair-pair-weighted aggregate; T(m, r) is its
   molecule-level decomposition; each G1 molecule contributes one D_m with equal weight. The pair-pair-weighted
   recombination serves only the implementation check (15 arrays x 2 roles within 1e-9). The 23-molecule means
   are not the published shares (0.30 and 0.08 in amide coupling, for example) and are not reported as such.
3. **Criterion D** is the full leave-one-molecule-out refit of section 5. If an owner label flips, the cohort is
   not redefined: the affected molecule-context entries are removed and criterion A is recomputed. Reported for
   each molecule and array: owner retained or flipped, shares before and after, product-forming pairs remaining.
   The published owner table is not modified.
4. **Phase 1** is the implementation check and section 5 (criteria A-D and the descriptives section 5 lists, all
   from the same T values), then stop with `RESULTS_MEMO_CORE.md`. Sections 7 and 8 are not run;
   `anatomy_depth.csv` is not opened. Phase 1 ends before any fresh-cohort yield read: the five fresh cohorts stay
   sealed with their design counts, structures and roles, condition identities, hashes and hypotheses; no yield
   or saved outcome of them is read. Opening any of them later is a separate decision.
5. `P2_Molecular_Owner/01_journal_submission/01_main_text_and_SI/Manuscript_Nature.docx` showed pre-existing byte drift
   before this execution; it is not used as an analytical input and will not be modified or reconciled in this
   task. Current state: modified 2026-10-04 10:29:39 -0400; 1,358,287 bytes; SHA-256
   17052bcde77cc4be969815067332f50263deab2a108e417ba6d8b69bd009f00c. The organisation map registers 1,185,722
   bytes, SHA-256 bb3faf1f5209a354f3a350ac2e10dd09fc72bbec3ec788297c082fd99c551e39.

## Note 1 — 2026-10-04 13:38 -0400, before any section 7 outcome was read (no rule, criterion or classification changes)

SHA-256 of this file above this heading: ab96def812c002a0ed60138f8609c2ea06bb55650b2af387635cdd3156828f7f

Phase 1 classified the redesign as FULL under section 6. Section 7 cannot change that classification; it decides
the application story (Figure 4, Extended Data). Phase 2 runs 7.5, 7.1, 7.3 primary (preference-only) and 7.3
secondary (additive), and 7.4 for the descriptive first crossing of the fixed trio. Not run in Phase 2: 7.2, 7.6,
7.7. The five fresh cohorts stay sealed. Where the section 7 text left a choice open, it is fixed here:

- 7.5: point estimates and record counts. Check first: the two roles recombined must reproduce the published
  pooled values (`Figure4a_history_depth.csv`: 3.993, 7.224, 8.699, 10.419, 12.439 per cent; 1,251 records).
- 7.1: targets, outcomes and per-set handling as in the frozen scorers (Set B drops conditions measured twice for a
  pair, Set A does not); training rows are the analysis rows with `allowed_for_model_fit`. Check first: at full
  depth the code must reproduce the saved per-target losses (`reserve_validation/record_transfer_losses.csv`,
  SHA-256 2f2e8316...; `cross_library/target_losses.csv`, SHA-256 3a4a1e33...) to 1e-9. Whole-array draws
  resample arrays and pool their sums, as `partner_external.summarise` does. The 844-target combination is added
  as a description.
- 7.3: s, slope-adjusted residuals and additive residuals use the product-forming pairs other than the target; the
  additive residual is P2's `train_residual` (yield minus the pair's row mean minus the training condition mean
  plus the training grand mean); ranking in both versions is the training condition mean plus the profile;
  `top_k` and `greedy_trio` are copied verbatim from `common.py`; the trio is fitted on all other pairs of the
  array, product-forming or not; a target is scored only if every condition of the trio and of all its owner- and
  partner-subset panels is observed; owner- and partner-panel best yields are averaged over their own subsets.
  The same 10,000 array draws serve every contrast and both versions.
- 7.4: rules of 7.3; a target enters if every condition of its panels at every K and of the trio is observed;
  pointwise intervals from 10,000 draws resampling owner (or partner) molecules within arrays, seed 20261008, the
  same draws at every K.

## Note 2 — 2026-10-04 13:40 -0400, before any section 7 outcome was read (implementation only)

SHA-256 of this file above this heading: 1ad9c32bf8c6dd5c4b4f0370b7213df666630687451eca61a7ad40b96fd092b4

7.4 asks for five K-pair subsets per target "by smallest SHA-256". Enumerating every subset is not feasible at
K = 6 (over two million subsets for the best-connected amide molecules), so subset j (j = 0 to 4) is the K history
partners with the smallest SHA-256 of "target subpair | j | partner canonical SMILES". The choice is fixed by the
identities alone, as before. 7.1 and 7.3 keep the stated rule (all two-pair subsets if at most 45, otherwise the 45
with the smallest SHA-256 of the sorted partner identities).

## Note 3 — 2026-10-04 13:54 -0400, before any 7.2, 7.6 or 7.7 outcome was read (implementation only)

SHA-256 of this file above this heading: 38676d11b75a7a95c7a32ab9a9bb6ab2c744d6ec9348a0c77aa7ac0fa4f22cf4

Phase 3 runs 7.2, 7.6 and 7.7, the remaining section 7 items, as supporting analyses; they cannot change the
section 6 classification. The five fresh cohorts stay sealed. Readings fixed where the text left a choice open:

- 7.2: all 910 targets are scored (no panel observability applies to a pattern endpoint); each target must have
  at least 8 observed conditions, as in the frozen transfer estimator (asserted, not used to drop targets);
  R2 for the owner and the partner side is reported with the pooled difference.
- 7.6: conditions sorted by the SHA-256 of the condition identifier, then alternately assigned starting with half
  A; the product-forming gate comes from the full array; shares use the published estimator within each half;
  no interval. BH|1 and BH|2 are reported with their published owners and their condition overlap.
- 7.7: a pair's slope is the slope of the pattern regression (the same least-squares fit on s); slopes are centred
  on the array's product-forming mean; the variance is the mean squared centred slope; the amplitude share of a
  role averages the products over all ordered pairs of pairs sharing a role molecule; no interval.
