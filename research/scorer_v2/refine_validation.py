"""Validation-only refinement: rule-backed immediate finish and fixed mixtures."""
import json,gzip
import numpy as np
from pathlib import Path
from experiment import ROOT,OUT,write,part_metrics,bundle_predict,encode_structured
def main():
 if (OUT/"refinement.json").exists():raise SystemExit("refinement frozen")
 rows=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
 states=[r["input"] for r in rows];y=np.array([[r["target"][k] for k in ("overall","reasoning_depth","move_concealment")] for r in rows],np.float32)
 split=json.loads((ROOT/"split.json").read_text());lookup={r["id"]:i for i,r in enumerate(rows)}
 va=np.array([lookup[i] for i in split["ids"]["val"]]);train=[lookup[i] for i in split["ids"]["train"]]
 pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
 finish=np.array([encode_structured(s)[-1]>.5 for s in states])
 assert np.all(y[[i for i in train+list(va) if finish[i]]]==1),"rule contradicts train or validation annotations"
 with gzip.open(OUT/"candidate_models.json.gz","rt") as f:arts=json.load(f)
 validation=json.loads((OUT/"validation.json").read_text())
 predictions={name:dict(np.load(OUT/(name+"_predictions.npz"))) for name in arts}
 audit={"kind":"validation_driven_second_stage","test_not_read":True,"rule":"legal immediate finish implies all three rubric scores=1; no solver","fixed_ensemble_weights":[[1,1],[2,1],[1,2]]}
 # Add deterministic gate to every method; retain the raw method for comparison.
 for name in list(arts):
  for key,a in list(arts[name].items()):
   gated=predictions[name][key].copy();gated[finish]=1
   gkey=key+"_finish_gate"
   arts[name][gkey]={"kind":"finish_gate","base":a};predictions[name][gkey]=gated
   validation[name]["validation"][gkey]=part_metrics(rows,y,gated,va,pairs)
  if name.startswith("fewshot"):continue
  v=validation[name]["validation"];validation[name]["selection"]=min(v,key=lambda k:(v[k]["macro_grade_mae"],v[k]["mae"]["overall"],k))
 # Fixed mixtures of validation winners from two independently fit weighting regimes.
 names=["merged_balanced","merged_unweighted"]
 selected=[validation[n]["selection"] for n in names]
 members=[arts[n][k] for n,k in zip(names,selected)]
 ps=[predictions[n][k] for n,k in zip(names,selected)]
 arts["merged_mixture"]={};predictions["merged_mixture"]={};validation["merged_mixture"]={"train_ids":split["ids"]["train"],"selection":None,"validation":{},"component_regimes":names,"component_methods":selected}
 for left,right in [(1,1),(2,1),(1,2)]:
  key=f"balanced{left}_unweighted{right}"
  arts["merged_mixture"][key]={"kind":"ensemble","members":[members[0]]*left+[members[1]]*right}
  pred=(ps[0]*left+ps[1]*right)/(left+right);predictions["merged_mixture"][key]=pred
  validation["merged_mixture"]["validation"][key]=part_metrics(rows,y,pred,va,pairs)
 v=validation["merged_mixture"]["validation"];validation["merged_mixture"]["selection"]=min(v,key=lambda k:(v[k]["macro_grade_mae"],v[k]["mae"]["overall"],k))
 # Re-select 2-shot method using mean across support episodes, never ensemble their supports.
 fsnames=[f"fewshot_{s}" for s in [101,202,303,404,505]]
 fmeans={k:float(np.mean([validation[n]["validation"][k]["macro_grade_mae"] for n in fsnames])) for k in arts[fsnames[0]]}
 fsel=min(fmeans,key=lambda k:(fmeans[k],k))
 for n in fsnames:validation[n]["selection"]=fsel
 options=[(validation[n]["validation"][validation[n]["selection"]]["macro_grade_mae"],validation[n]["validation"][validation[n]["selection"]]["mae"]["overall"],n,validation[n]["selection"]) for n in names+["merged_mixture"]]
 _,_,regime,method=min(options)
 write(OUT/"selection_stage1.json",json.loads((OUT/"selection.json").read_text()))
 write(OUT/"selection.json",{"selected_regime":regime,"selected_method":method,"criterion":"validation macro-grade overall MAE, then overall MAE; merged regimes only","test_used_for_selection":False,"fewshot_method":fsel,"fewshot_validation_means":fmeans})
 write(OUT/"refinement.json",audit);write(OUT/"validation.json",validation)
 for n,ps in predictions.items():np.savez_compressed(OUT/(n+"_predictions.npz"),**ps)
 with gzip.open(OUT/"candidate_models.json.gz","wt") as f:json.dump(arts,f,separators=(",",":"))
 for n in names+["merged_mixture","old100_balanced"]:
  k=validation[n]["selection"];m=validation[n]["validation"][k]
  print(n,k,"val_macro",m["macro_grade_mae"],"val_overall",m["mae"]["overall"],flush=True)
 print("CHOICE",regime,method)
if __name__=="__main__":main()
