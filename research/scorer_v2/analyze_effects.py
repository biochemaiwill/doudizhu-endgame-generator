"""Diagnostic comparisons after the frozen test evaluation; does not select models."""
import json
import numpy as np
from experiment import ROOT,OUT,part_metrics,write
def main():
 rows=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
 ids={r["id"]:i for i,r in enumerate(rows)};split=json.loads((ROOT/"split.json").read_text())
 te=np.array([ids[i] for i in split["ids"]["test"]])
 y=np.array([[r["target"][k] for k in ("overall","reasoning_depth","move_concealment")] for r in rows])
 pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text());out={}
 for model in ("ridge_counts_1","ridge_counts_1_finish_gate","structured_mlp","role_ordinal_rank","extra_depthNone_leaf2"):
  out[model]={}
  for regime in ("old100_balanced","merged_balanced","merged_unweighted"):
   pred=np.load(OUT/(regime+"_predictions.npz"))[model]
   out[model][regime]=part_metrics(rows,y,pred,te,pairs)
 write(OUT/"controlled_comparisons.json",out)
if __name__=="__main__":main()
