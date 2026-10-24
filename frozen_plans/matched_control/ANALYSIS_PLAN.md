# Matched profile-substitution analysis — recorded design

28 September 2026. The machine-readable `analysis_config.json` records the UTC time and numerical rules before the new comparisons are computed. This is an additional retrospective analysis developed with knowledge of the existing manuscript results. It is not a prospectively registered experiment. Published data, baseline scientific modules, manuscript text and the frozen ten-plate protocol are not changed.

## Question and population

Does a building block's own experimental history give better condition choices than another building block's history of matched size and condition coverage? Analyze both partner roles separately in all fifteen recurring-building-block arrays; never select the more predictive role. Roles have different chemical meanings across arrays and are named in every output.

Use the existing analysis rows, duplicate-condition exclusion, three repeats of five substrate-pair folds (`stream(array, 'transfer', rep)`), training condition means, single-role residual profiles with shrinkage 4, and three-condition panel size. Each evaluation target must have measured yields for every condition in its array. This keeps the recommendation domain unchanged and all own, donor and fixed panels equally observable. No target outcomes enter the training quantities, matching or recommendations.

A mask/count-only feasibility check found 949 fully measured substrate pairs. Across two roles and three repeats they give 5,694 target-role-repeat records; 5,013 have a known identity in training. Requiring exact equality of the entire per-condition count vector leaves only 1,279 records and no matches in some array-role groups. No new recommendation effects were computed for this feasibility check. The primary tolerance rules below were therefore selected before computing effects; exact count vectors remain a declared sensitivity analysis. Do not loosen these rules in response to a result.

## Matching and controls

All donors are different molecular identities from the same array, role and training fold, with at least one training pairing. History depth is the number of training substrate pairs containing that molecule. Let the observation-count vector record, for each condition, the number of these pairings with a measured yield. Support comprises conditions with positive counts.

Primary donors must have exactly the same history depth as the target identity; support Jaccard at least 0.95; an absolute difference of at most one in the number of supported conditions; a maximum difference of one in per-condition counts; and the sum of absolute count differences divided by (history depth × array condition count) at most 0.05. These rules closely match history amount and support but do not assert identical per-condition shrinkage. Report actual balance, not just the thresholds.

The prespecified controls are:

1. **Unmatched diagnostic:** every alternative identity in the same array, role and fold. This is exhaustive averaging rather than the original 100 random draws and uses the complete-target population.
2. **Primary:** the depth/coverage rules above, averaging every eligible donor with equal weight.
3. **Strict sensitivity:** exact history depth and exact per-condition count vector.
4. **Activity secondary:** primary donors additionally within five yield percentage points of the target identity's training-only mean activity. Calculate activity by averaging each training pair's mean yield and then averaging these pair means within identity.
5. **Structure secondary:** among primary donors with Morgan radius-2, 2,048-bit Tanimoto similarity at least 0.5, take the nearest structural neighbour; average exact similarity ties (tolerance 1e-12). Report coverage and similarity. If no donor meets the threshold, report no match; do not silently use distant molecules. This is a nearest-analogue challenge, not matching a substitute to the self-similarity of 1.
6. **History-depth sensitivity:** primary comparisons restricted to fold records with at least two prior pairings per identity.

Donor matching never uses a target's response profile, held-out yield, observed gain, selected conditions or the previously chosen stronger role. Activity uses training yields only. Molecular structure is fixed before scoring. Partner composition, experimental time and batch are not matched; no result establishes a context-free intrinsic molecular property.

## Scoring and aggregation

For each fully observed target and eligible fold, compare its best measured yield over the own-history top three with the average of its best measured yields over each eligible donor-history top three. Both add the same training condition mean to the chosen single-role profile. Also record each panel's gain over the mean-ranked training trio on exactly the same targets and folds. Enumerate donors rather than sample them to remove donor Monte Carlo error.

Average donors within target-fold, then valid repeats within target, then targets with equal weight. A target contributes with one or more valid repeats; report the number with one, two or three, and a prespecified all-three-repeat summary. Never treat donor comparisons or repeat folds as independent experimental samples.

Primary reporting separates p1 and p2, both by array and pooled across arrays; no selected-role combined headline. Report the target-weighted pooled mean and a 95% percentile interval from 10,000 whole-array bootstrap draws. Provide a 5,000-draw within-array role-molecule bootstrap as auxiliary uncertainty conditional on the stored predictions. Neither interval refits models. Also show the equally weighted array mean descriptively. Small clusters and absent array-role matches are visible in the outputs.

To compare controls, use common target-fold records before averaging repeats. In particular compare primary with strict, activity and structure only on their intersection, and unmatched with primary on their intersection. Do not attribute differences between marginal summaries to matching alone.

## Missingness, attrition and interpretation

Report all arrays and roles, including zero matches, along with total targets, fully measured targets, training-seen targets, matched targets, donor counts and the number of eligible repeats. Partial-target evaluation is outside this analysis. These results concern the fully measured, matchable subset, not all original 1,855 substitution targets or all 212,398 reaction records.

Positive results support additional predictive value of the target's own history under the stated matching. Null or negative results are reported with equal prominence. This retrospective analysis is not a Nature acceptance test, a causal identification experiment or a substitute for the frozen prospective study.

## Verification and deliverables

Before scoring, record hashes of the data, baseline modules, configuration and this plan. Verify that changing held-out yield values with the missingness mask fixed cannot change profiles, donors or recommendations; confirm role-label permutation invariance; test synthetic equal-information null and identity-dependent cases; check matching constraints, donor exclusions, complete-panel scoring and target-level aggregation. An independent review examines the implementation.

Deliver the script, frozen configuration/plan, fold assignments, candidate/donor-level source data, attrition and matching balance, target and array summaries, paired-control comparisons, a readable results report and a standalone scientific figure. No prospective outcomes are manufactured or inferred from empty bench sheets.
