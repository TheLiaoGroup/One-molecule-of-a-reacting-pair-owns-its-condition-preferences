# Pre-registration: does the owner's record win on pairs outside the arrays, across thirteen reaction systems?

Frozen 1 October 2026, before any endpoint is computed on the pairs described below. Code:
`03_CODE/13_cross_library/cross_library_validation.py`; its SHA-256, this file's, the sealed data file's
and the outcome-free manifest's are in `PREREGISTRATION.sha256`.

## Claim

The partner that owns a pair's condition preference in the analysed arrays also owns it outside them.
A first sealed test (`04_ANALYSES/11_reserve_validation/`) established this on 309 pairs in six arrays.
This one asks the same question of a different and larger population covering thirteen of the fifteen
reaction systems, including seven the first test could not reach: Suzuki, C–N (Pd ligand), SNAr, Kumada,
Chan–Lam, Grignard and Sonogashira.

## Data

`02_DATA/04_SEALED/never_opened_cross_library.csv.gz`: 30,556 rows, 535 substrate pairs, thirteen
arrays. These are measurements the platform recorded on this paper's own condition libraries but which
fall outside its rectangular arrays, so they were never part of any array and have entered no analysis
in this work. Each pair covers at least eight conditions of its array's library, and both of its
molecules carry at least two distinct training partners. Panels are incomplete by construction — a
median of 26 to 71 conditions of 32 to 95 — which is why the yield-decision endpoint is not attempted
here; it is not evaluable without complete panels.

No (array, pair) of these rows occurs in `DATASET_analysis.csv.gz`, `DATASET_validation_sealed.csv.gz`,
`measurements.csv.gz`, `analysis_measurements.csv.gz`,
`development_seen_analysis_fit_excluded.csv.gz` or `never_analysed_reserve.csv.gz`; the installer
asserts this and the scoring run re-asserts it.

Per array: amide coupling 169 pairs, Suzuki 91, C–N (Pd ligand) 55, Heck 50, Chan–Lam 41, Kumada 35,
SNAr 24, Grignard 18, Ullmann 15, Hiyama 14, reductive amination 13, Sonogashira 5, esterification 5.

## Already seen before freezing

Counts only: pairs, rows and conditions per array, and each array's mean yield over these rows. No
single-role profile, no loss and no endpoint has been computed on them. The power emulation below uses
analysis pairs alone.

## Endpoints

Both reuse the estimator frozen for the archived electrophile comparison
(`03_CODE/11_partner_asymmetry/partner_external.py`), with the owning building block in place of the
aryl electrophile. That file is imported, not copied, and is unchanged by this analysis.

Single-role profiles are fitted on the analysis rows of each array alone. A sealed pair's pattern is
its ordinary-least-squares residual on an intercept and the training condition means over the
conditions it covers; the owner-only and partner-only predictions are projected the same way. A pair
enters with at least eight finite cells.

### Primary, a test — the pooled difference

D = (ΣSSE_partner − ΣSSE_owner) / ΣSST, pooled over all scorable pairs. Uncertainty: 2,000 draws,
seed 20261007, resampling (i) owning building blocks within arrays and (ii) whole arrays.

### Secondary, an estimate — how many arrays

The number of the thirteen arrays whose own D exceeds zero, with the one-sided binomial probability of
that many or more under a coin. Reported with no pass line.

## Power

Emulated on held-out analysis pairs at these target counts, through the same estimator. Each array
holds out as many pairs as the sealed set supplies, capped at 40% of the array, so training is smaller
than in the real test and these figures are a lower bound.

| Endpoint | Median value | Power | Role |
|---|---|---|---|
| Pooled D | 0.161 | 1.00 under both resampling schemes | **test** |
| Arrays favouring the owner | 10 of 12 | 0.64 at *p* < 0.05 | estimate |

200 replicates, median 309 emulated targets against 535 real ones. Only the pooled difference reaches
80%, so only it is a test; the per-array count is registered as an estimate and is reported whichever
way it falls.

## Decision rules

- **D > 0 and both lower bounds > 0:** "Outside the arrays as well, the owner's record predicted a
  pair's condition preference better than its partner's, across thirteen reaction systems."
- **D > 0, owner-cluster lower bound > 0, whole-array lower bound ≤ 0:** the same sentence restricted
  to these pairs, with the per-array values reported.
- **Otherwise:** ownership is reported as a property of the measured arrays and of the first sealed
  test, and this population is reported as not reproducing it.

The per-array count is reported as a count with its binomial probability in every case, and is not
described as a confirmation or a failure.

## What is not claimed

These wells were recorded on the same platform and in the same campaigns as the analysed arrays. They
were never opened by this paper, which is what makes them a test of the rule rather than of a new
experiment. Their release is limited to the 535 pairs scored here.

## Amendments

Any change after the hashes below is recorded here with its time and reason, and states whether any
outcome had been read.
