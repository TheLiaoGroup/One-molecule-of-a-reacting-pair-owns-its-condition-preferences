"""Figure 5: approved transfer comparisons and retained partner-depth curves."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import v6_core as c
import figure4 as drawing
import pandas as pd
from figure4_bootstrap_data import write_seta_bootstrap


def main():
    drawing.OUT=c.SOURCES
    with drawing.style():
        fig,observation,tables=drawing.draw5()
        folds=pd.read_csv(c.ROOT/"05_DATA/Analysis/figure_display/Fig4a_folds.csv")
        assert not folds.duplicated(["screen","subpair"]).any()
        mapping=folds.set_index(["screen","subpair"]).fold
        records=tables["C_retained_depth_records"]
        assert all(int(mapping.loc[(row.screen,row.target_pair)])==int(row.fold)for row in records.itertuples())
        drawing.save(fig,"Figure5",observation,tables)
    bootstrap=write_seta_bootstrap(c.SOURCES)
    for name in bootstrap:
        path=c.SOURCES/name
        c.OUTPUT_RECORDS[name]=(c.ROOT/"04_SOURCE_DATA/Figure5"/name,path)
    counts=tables["C_selection_counts"].set_index("screen").common_targets.sort_index()
    c.write_json("figure5_selection.json",dict(rule_sha256=c.digest(c.ROOT/"07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"),
        threshold_common_targets=20,common_targets=310,array_counts=counts.astype(int).to_dict(),
        pooled_arrays=counts.index.tolist(),omitted_thin_arrays=counts[counts.lt(20)].index.tolist(),
        selected_arrays=[dict(screen=key,targets=int(value))for key,value in counts[counts.ge(20)].items()]))
    return {"panels":{"A":{"Set_A_pairs":309,"Set_B_pairs":535},
                      "B":{"equal_depth_intervals":observation["equal_depth_intervals"]},
                      "C":{"fixed_common_targets":310}},"bootstrap":bootstrap}


if __name__=="__main__":
    c.cli(main,5)
