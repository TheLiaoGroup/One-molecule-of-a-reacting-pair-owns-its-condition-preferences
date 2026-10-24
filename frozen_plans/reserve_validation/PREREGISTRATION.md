# Pre-registration: does condition-preference ownership hold on pairs this paper has never analysed?

Frozen 1 October 2026, before any endpoint is computed on the reserve pairs. Code:
`03_CODE/12_reserve_validation/reserve_validation.py`; its SHA-256, this file's, and the sealed
data file's are in `PREREGISTRATION.sha256`.

## Claim

The two measurements this paper reports from its analysed arrays — that one building block owns the
larger share of a pair's condition preference, and that the owner's record predicts how it will behave
with a partner it has not met — hold on substrate pairs that entered no analysis in this paper.

## Data

`02_DATA/04_SEALED/never_analysed_reserve.csv.gz`: 38,300 rows, 442 substrate pairs, seven of this
paper's fifteen arrays, every pair measured against the whole of its array's condition library. The
rows are the subset of the corpus's 39,960-row reserve evaluation partition
(`allocation = "reserve: substrate not sampled into the analysis set"`, source SHA-256
`ecec4bdcdb718533dc7f44ab8b96c5ec53b9d8e74eb38050f7f73d54944cf620`) that falls in those seven arrays;
the three process-development C–N arrays are excluded here as they are throughout the paper. Every row
carries `allowed_for_model_fit = False` and the reader refuses it for fitting.

The installer asserted, and this run re-asserts, that no (screen, pair) of these rows occurs in
`DATASET_analysis.csv.gz`, `DATASET_validation_sealed.csv.gz`, `measurements.csv.gz`,
`analysis_measurements.csv.gz` or `development_seen_analysis_fit_excluded.csv.gz`. The 11,249 archived
measurements this paper already scores come from the corpus's never-release partition and share no row
with the reserve partition.

Per array: Acylation 247 pairs over 94 conditions, Heck 52 over 90, Reductive-amination 48 over 47,
Ullmann 47 over 95, Hiyama 31 over 92, C–O competitive 15 over 92, Esterification 2 over 95.

## Already seen before freezing

Counts and descriptive quantities only, listed so that the freeze is honest:

- Pairs, molecules and conditions per array; which of each pair's two molecules already occur in the
  analysis set (383 of 460 reserve pairs are new combinations of molecules that both occur there, 8
  pairs have two new molecules); how many earlier partners each carries in the analysis set.
- Each array's mean yield over its reserve rows, and the number of reserve pairs reaching 10% yield
  under at least one condition.
- Power emulations run entirely on analysis pairs (below). No reserve yield enters them.

No single-role profile, no share, no recommendation and no endpoint has been computed on these rows.

## Endpoints

Each endpoint reuses the estimator that produced the corresponding main-text number; none is
reimplemented here.

### E1 — ownership, re-estimated on reserve pairs alone (estimate, no pass line)

For each array with at least 50 product-forming reserve pairs, the method-of-moments shares of
`ctb/anatomy.py` are computed on the reserve pairs only, after the same product-forming gate
(maximum yield at least 0.10) and the same slope-adjusted pattern. Reported: which role carries the
larger share, the margin owner minus partner, and a 95% interval from 2,000 molecule resamples
(`anatomy.molecule_weights`). The gate is outcome-dependent and pre-specified; it exists because pairs
that form no product are correlated by construction once condition means are removed, which is why the
main text applies it too.

### E2 — whose record predicts a new combination (confirmatory test)

Single-role profiles are fitted on the analysis rows of each array alone, by
`03_CODE/11_partner_asymmetry/partner_external.py` — the estimator frozen for the paper's external
test, with the owning building block in place of the aryl electrophile. A reserve pair is a target when
both of its molecules carry at least two distinct training partners and it has at least eight finite
cells. Its pattern is the OLS residual on [1, training condition means]; the owner-only and
partner-only predictions are projected the same way.

Statistic: D = (ΣSSE_partner − ΣSSE_owner) / ΣSST, pooled over targets. Uncertainty: 2,000 draws,
seed 20261004, resampling (i) owning building blocks within arrays and (ii) whole arrays.

### E3 — the three-condition decision (estimate, no pass line)

For each reserve pair, `validation_core.recommend` ranks the array's conditions by the training mean
plus both fitted profiles and takes the top three; `validation_core.score_panel` compares the best
measured yield among them with the best among the frozen complementary trio. Reported over all
eligible pairs and over product-forming pairs, with 95% intervals from resampling building blocks
within arrays (`validation_core.summarize_effects`, 10,000 draws, seed 20260928).

## Power

Emulated entirely on analysis pairs, at the reserve target counts, through the same frozen estimators.
Each array holds out as many pairs as the reserve supplies (capped at 40% of the array) and trains on
the rest; the real test trains on the whole array and takes the reserve pairs as extra targets, so
training here is smaller and these figures are a lower bound.

| Endpoint | Targets | Median effect | Power | Role |
|---|---|---|---|---|
| E1 ownership, Acylation | 217 product-forming pairs | — | 0.28 | estimate |
| E2 record transfer | 233 | D = 0.158 | 1.00 owner clusters, 0.76 whole arrays | **test** |
| E3 decision | 344 | +1.04 points | 0.29 | estimate |

200 replicates for E2 and E3, 50 for E1 (each with 400 inner draws); the scoring run
uses the full draw counts given above. Only one array supplies at least 50 product-forming reserve
pairs, so E1 is a single-array estimate by construction.

Within E2 the two resampling schemes are not equally powered: resampling owning building blocks within
arrays reaches 1.00, resampling whole arrays 0.76. The powered claim is therefore the
one the owner-cluster bound licenses; the whole-array bound is reported as the stricter secondary.

E2 is therefore the confirmatory test. E1 and E3 do not reach 80% and are pre-registered as estimates:
they are reported with intervals and per-array signs, and no outcome of either is described as a
confirmation or a failure.

## Decision rules

Each rule states a direction; an interval that merely excludes zero is not sufficient.

**E2, the test.**

- *D > 0 and both lower bounds > 0*: "In combinations this paper never analysed, the owning building
  block's record predicted condition preference better than its partner's."
- *D > 0, owner-cluster lower bound > 0, whole-array lower bound ≤ 0*: the same sentence restricted to
  these combinations, with the per-array signs reported.
- *Otherwise*: ownership is reported as a property of the analysed arrays only, and the main text says
  that it did not carry to the never-analysed pairs.

**E1 and E3, the estimates.** Reported as point estimates with 95% intervals, per array for E1 and
pooled with the product-forming split for E3. The paper may say which way they point and must give the
interval; it may not call either one a confirmation, and may not drop either one for pointing the wrong
way. Both are reported whatever they show.

## What is not being claimed

These wells were run on the same platform, in the same campaigns, as the analysed arrays; they were
never opened by this paper, which is what makes them a test of the rules, not of a new experiment.
The reserve partition is also listed as planned-public by two sibling manuscripts. Any sentence written
from this test says that the rows entered no analysis in *this* paper, not that they have never been
analysed.

## Amendments

Any change after the hashes below is recorded here with its time and its reason, and states whether any
outcome had been read.
