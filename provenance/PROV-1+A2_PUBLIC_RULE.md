# Provenance matching rule PROV-1+A2

This is a versioned rule. PROV-1 was frozen before any count was computed. Amendment 1 was made before the first count, and Amendment 2 was adopted after the first PROV-1 run (section 6). The final counts use PROV-1+A2.

The rule decides which of the 244,179 measurements released with this manuscript exactly match records of the published amide-coupling collection of Zhang *et al.* (2025). The counts serve disclosure only. No analysis, figure or release scope depends on them.

`06_CODE/Release/provenance_overlap_public.py` implements this rule and nothing else. Its results are the two files next to this one: `P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json` (counts) and `P2_RECORD_PROVENANCE.csv` (one row per released measurement).

## 1. Datasets

| Name | Records | Content |
|---|---:|---|
| P2 | 244,179 | The released measurements of this manuscript, in four release units: A1, 177,039 (the fifteen arrays without the duplicated Hiyama condition); A3, 74 (the duplicated Hiyama condition); SA1, 36,499 (the 420 released pairs of Set A, the first out-of-array evaluation set); SB1, 30,567 (the 535 pairs of Set B, the second). |
| Zhang | 47,015 | `data/all_HTE_with_condition.csv` of github.com/aichemeco/amide_coupling at commit `f6fdbe283fc24af53389aab8dcba5e46386ac9c0`: Git blob `778db5176ed6a1b3b11cbb80da4eacbd40ea74c5`, 17,657,403 bytes, SHA-256 `5085e1ae50ee26173bd170a4b10a99c4e3101a46b706d115e425aff17c091b41`. The script stops unless the file it reads, decompressed if gzip-compressed, has this SHA-256. |

The condition dictionary is the condition table of the release, 1,192 conditions, whose reaction types are the libraries below. The project copy, `05_DATA/Release/00_RAW_IMMUTABLE/DATASET_conditions.csv.gz`, also holds conditions used only by unreleased records; with or without them, every count and flag is the same. The roles of the two partners in each screen are given by the table `ROLE` in `06_CODE/Core/00_common/ctb/history.py`.

## 2. Record key

A record's key has five parts: library, condition label, partner 1, partner 2 and yield. Two records match only when all five are equal. A record without a key matches nothing.

1. **Library.** The reaction type (`reaction_type` of the condition dictionary) to which the record's condition resolves under section 3.
2. **Condition label.** The record's condition label within that library, under section 3.
3. **Partners.** The two partners as RDKit canonical isomeric SMILES, `Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)`, placed in the library's role order under section 4. A SMILES that RDKit cannot parse gives no key. The RDKit version is recorded with the counts.
4. **Yield.** The stored yield, a fraction of 1 in both datasets, parsed as a floating-point number. It is rounded with Python `round` to nine decimal places and then to six, and negative zero is written as zero.

Rounding to nine places first joins floating-point representations of one value that would fall on opposite sides of a six-place boundary, such as 0.2700134999999999 and 0.2700135.

**Guard.** Take every pair of records from two datasets whose keys agree under this rule but differ under a single rounding to six places. Their stored yields must differ by at most 10⁻⁹, or the run stops.

## 3. Condition resolution

- **P2.** The library is the reaction type paired with the record's `rxn_class` in the condition dictionary, and the label is `condition_id`. The record's own reagent slots 1–6, `solvent_id`, `temp_id` and `time_id` must equal that dictionary row; empty equals empty, and numbers are compared as numbers. Otherwise the record has no key.
- **Zhang.** The library is the reaction type of `rxn_class` "Acylation", and the label is `'C'` + `condition_id`. The label must exist in that library, and its composition must agree through the code bridge below. Otherwise the record has no key.

**Code bridge.** Zhang records name their components by codes (`Activation_ID`, `Additive_ID`, `Base_ID`, `solvent_id`). Each code is mapped to a dictionary name by elimination over the labels present in both datasets:

1. A code's candidates are the intersection, over every label that uses it, of that label's dictionary names. These are the reagent names for activation, additive and base codes, and the solvent name for solvent codes.
2. Repeat until nothing changes:
   - a code with exactly one candidate is assigned it, and a code with none is assigned "none";
   - each assigned name is then removed from the candidates of the unassigned codes of the same component role (activation, additive, base or solvent).
3. A label agrees when two conditions hold:
   - its codes that are not "none" can be paired one to one with its dictionary reagent names, each code with a name among its candidates. Codes left with several candidates are allowed whenever such a pairing exists;
   - its solvent code is assigned the dictionary solvent.

## 4. Role order

The key places the partners in the library's order (`sub_1`, `sub_2`), whose roles in each screen are given by `ROLE`.

- **P2.** Records are stored in this order.
- **Zhang.** Records are taken in stored order (`sub_1_smiles`, `sub_2_smiles`). This holds only if two shares both reach 95%: the distinct `sub_1` molecules matching the role-1 SMARTS of the amide library, and the distinct `sub_2` molecules matching its role-2 SMARTS. Otherwise no Zhang record has a key.
## 5. Counting: one to one

For each key, let n(P2) and n(Zhang) be the numbers of records carrying it. Records are paired one to one, never by set membership: the j-th record of a key in one dataset is paired with the j-th record of that key in the other dataset when it holds at least j of them.

The overlap is the sum, over keys, of min(n(P2), n(Zhang)). A record without a key matches nothing.

**Per-record flags.** Within a key, P2 records are taken in release order: by unit (A1, A3, SA1, SB1), then by row. The first min(n(P2), n(Zhang)) records of a key are flagged `matches_zhang2025`, and `match_rule_version` is `PROV-1+A2`. The sum of the flags reproduces the overlap count.

A match means that two records are identical under this rule. It does not establish that they come from one physical experiment, and a record without a match is not thereby a first-time experiment.

## 6. Version history

- **PROV-1** was frozen on 2026-10-05, before any count was computed. Before freezing, only the datasets' formats and condition dictionaries had been inspected; no yields were compared between datasets.
- **Amendment 1** was made before the first count. A check of the rule's mechanics, which compared no yields, found a code-bridge defect:
  - every code-bridge candidate set contained "none", so no reagent code could reach a single candidate;

  Section 3 gives the amended code bridge. The complete internal record retains the other dataset-specific amendments.
- **Amendment 2** was adopted after the first PROV-1 run. It is a post-first-run correction of two documented mechanical matching defects; it was not specified before counting.
  - **Cross-role elimination.** Elimination in the code bridge crossed component roles. DMAP is an additive in some amide conditions and the base in C46 (BOPCl; DMAP), so both C46 codes were mapped to BOPCl, and the 462 Zhang records under C46 had no key. Yet the collection's own condition SMILES for C46 lists BOPCl, DMAP and DMF, exactly as the dictionary does. Elimination now stays within a component role (section 3).
  - **Rounding boundary.** A single rounding to six places separated floating-point representations of one value. The yield is now rounded to nine places first, under the guard of section 2.
- **Internal record.** The complete rule history and earlier counts are retained internally and are available to editors and reviewers on request.
