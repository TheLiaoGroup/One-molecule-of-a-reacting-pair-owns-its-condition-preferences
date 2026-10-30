# Figure 1 draft 3 display rule

Authority: author-approved Figure 1 brief (draft 3, 2026-10-28).
Scope: Figure 1 wireframe only. Draft 2 is retained in draft2_fig1_archive.
No manuscript, Figure 2-5, official source, analysis, lock or remote edits.

## Definition frozen before the new aggregate calculation

- P1 remains the shared Buchwald-Hartwig|1 pair: 2-amino-4,6-dimethylpyridine
  with 2-chloro-4-cyanopyridine. Reuse the existing residual preference
  arrays in aminations_payload.json; do not refit or change eligibility.
- The halide record averages all six nonshared amine_rows, each holding
  P1's halide fixed. The amine record averages all nine nonshared
  halide_rows, each holding P1's amine fixed. Explicitly exclude P1 by
  pair identity and shared flag. Do not select individual examples.
- At each condition take the arithmetic mean of finite existing values,
  with equal weight per available pair. Retain the contributor count.
  An all-missing condition remains missing. Do not insert zero, impute,
  clip, reweight, renormalize or recompute residuals before averaging.
- Sort all 95 conditions by P1's preference descending, ties by retained
  original condition index, missing P1 last in retained order. Use this
  one permutation for the reference, both record strips and both steps.
- Split the sorted 95 positions into ten contiguous segments with
  numpy.array_split: first five have 10 positions, last five have 9.
  Each step is the finite arithmetic mean of that record's existing
  condition averages in its segment. Empty segments remain gaps. Do not
  change bins after inspecting results or fit a trend line.
- Compute each record's Pearson r against P1 on the finite unbinned
  condition intersection. Report sample conditions and member pairs.
  These are not correlations of the ten displayed step values, nor the
  full-experiment pairwise means 0.27/0.07. Neither r enters the text.
- Retain gray-white-gold at -30 to +30 pp, with existing display-only
  saturation and hatched missing cells. Both step plots use -30 to +30
  pp, identical physical axis height and a visible zero baseline. Do not
  rescale each record independently or add smoothing.
- If the halide-record r is below 0.4, render and report it unchanged;
  never select other members, change the definition or substitute P1.

## Access history and freeze evidence

The residual rows, existing analyses and earlier selected-pair values
were already viewed in draft 1/2. This is an author-directed retrospective
display revision, not a blinded or prospective scientific test. Save this
rule and a separate UTC/SHA-256 freeze before the first new aggregate
calculation. Preserve the previous rules/results and any failure evidence.

## Layout and identity

- Editable layered SVG/PNG, 180 x 150 mm, existing native chemical scale.
  Main story area is x=5..138, y=28..133 mm. The target node occupies
  x=78..138, y=37..127 mm (about 39% of that area), unframed and white.
- Show P1 only once as the sorting reference, plus two aggregate strips
  and their ten-segment steps. Remove all old selected-pair strips and
  their 0.18/-0.04 annotations. Keep the true full-group 0.27/0.07 separate.
- Restore the label amine's record. Show the focal molecule beside each
  record. The central unmeasured target keeps the previous target pair,
  whose amine differs from P1's. It illustrates the query, not a claim
  that both plotted histories belong to that central pair. The caption
  states this distinction; arrows end at the generic condition question,
  not at either target molecule. Do not call measured P1 never measured.
- Connect literature labels to known molecules, the amine route and the
  condition question. Keep right-hand route/results, top reaction and
  bottom preference scale. Do not restore the duplicated plate.
- At most 15 story annotations, at most five words per annotation,
  excluding native chemical atom glyphs and quantitative axis ticks.
  Keep 7/5 pt text and native 5.25 pt subscripts. Split each record label
  and its pair count/r into two adjacent annotations to meet the limit.
- Put the updated Figure 1 caption in a Figure-1-only draft caption file;
  do not regenerate the six text drafts during this explicitly limited round.

## Execution checklist

- [x] Archive draft 2 Figure 1 and verify byte identity; official preflight.
- [ ] Save UTC/SHA-256 freeze; compute records and contributor counts.
- [ ] Draw new SVG/PNG and Figure-1-only caption, preserving other drafts.
- [ ] Independently recompute means, r, column order, bins and rendered
      values; test inclusion-of-P1 and independent-scaling failures.
- [ ] Inspect actual PNG at print size/200%, measure critical gaps,
      verify official and other-draft hashes, append report section 10.
- [ ] Stop for author review.
