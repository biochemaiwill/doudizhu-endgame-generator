"""Combine predeclared validation candidates; final test stays sealed."""
import json,numpy as np
from pathlib import Path
from experiment import ROOT,OUT,write,part_metrics,encode_structured
def main():
 if (OUT/"selection_final.json").exists():raise SystemExit("final choice already frozen")
 rows=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
 idmap={r["id"]:i for i,r in enumerate(rows)}
 split=json.loads((ROOT/"split.json").read_text());va=np.array([idmap[i] for i in split["ids"]["val"]])
 y=np.array([[r["target"][k] for k in ("overall","reasoning_depth","move_concealment")] for r in rows],np.float32)
 pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
 finish=np.array([encode_structured(r["input"])[-1]>.5 for r in rows])
 v=json.loads((OUT/"validation.json").read_text());candidates={};preds={}
 for regime in ["old100_balanced","merged_balanced","merged_unweighted","merged_mixture"]:
  method=v[regime]["selection"];key=regime+"/"+method
  p=np.load(OUT/(regime+"_predictions.npz"))[method]
  candidates[key]={"type":"builtin","regime":regime,"method":method};preds[key]=p[va]
 prior=json.loads((ROOT.parent/"scorer_v2_prior_trial/validation.json").read_text())
 assert prior["val_ids"]==split["ids"]["val"] and prior["train_ids"]==split["ids"]["train"]
 for mode in ["board","structured"]:
  p=np.array(prior["models"][mode]["prediction"])
  for gate in [False,True]:
   key="tabpfn_v2/"+mode+("/finish_gate" if gate else "")
   q=p.copy()
   if gate:q[finish[va]]=1
   candidates[key]={"type":"tabpfn","features":mode,"finish_gate":gate};preds[key]=q
 # Fixed convex combinations, not a fitted meta-regression on twelve labels.
 prior_key="tabpfn_v2/structured/finish_gate"
 bases=list(k for k in candidates if candidates[k]["type"]=="builtin")
 for base in bases:
  for prior_weight in [.25,.5,.75]:
   key=f"mixture/{base}/prior{prior_weight}"
   candidates[key]={"type":"mixture","components":[{"candidate":base,"weight":1-prior_weight},{"candidate":prior_key,"weight":prior_weight}]}
   preds[key]=preds[base]*(1-prior_weight)+preds[prior_key]*prior_weight
 scored={}
 for key,p in preds.items():
  full=np.zeros_like(y);full[va]=p
  scored[key]=part_metrics(rows,y,full,va,pairs)
 selected=min(scored,key=lambda k:(scored[k]["macro_grade_mae"],scored[k]["mae"]["overall"],k))
 write(OUT/"selection_final.json",{"selected":selected,"spec":candidates[selected],"candidates":candidates,"validation":scored,"criterion":"minimum validation macro-grade MAE then overall MAE","test_used_for_selection":False,"note":"Includes validation-driven rule gate and fixed prior mixtures; this is development, not a preregistered final benchmark."})
 print("FINAL_SELECTED",selected)
 for k in sorted(scored,key=lambda k:scored[k]["macro_grade_mae"])[:8]:print(k,scored[k]["macro_grade_mae"],scored[k]["mae"]["overall"])
if __name__=="__main__":main()
