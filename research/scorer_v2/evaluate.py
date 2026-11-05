"""One-shot reporting after selection_final.json was written."""
import sys,json,gzip,hashlib
from pathlib import Path
import numpy as np
from experiment import ROOT,OUT,write,part_metrics,bundle_predict
from models import encode_structured
sys.path.insert(0,str(ROOT.parent/"scorer_v1"))
from scorer import predict_export
def main():
 if (OUT/"test_report.json").exists():raise SystemExit("refuse re-evaluate frozen report")
 selection=json.loads((OUT/"selection_final.json").read_text())
 assert selection["test_used_for_selection"] is False
 rows=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
 states=[r["input"] for r in rows];lookup={r["id"]:i for i,r in enumerate(rows)}
 split=json.loads((ROOT/"split.json").read_text());idx={p:np.array([lookup[i] for i in ids]) for p,ids in split["ids"].items()}
 y=np.array([[r["target"][k] for k in ("overall","reasoning_depth","move_concealment")] for r in rows],np.float32)
 pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
 prior=json.loads((ROOT.parent/"scorer_v2_prior_trial/predictions_test.json").read_text())
 assert prior["ids"]==split["ids"]["test"]
 finish=np.array([encode_structured(r["input"])[-1]>.5 for r in rows])
 candidates=selection["candidates"];selectedpred={}
 def get(key):
  if key in selectedpred:return selectedpred[key]
  c=candidates[key]
  if c["type"]=="builtin":
   p=np.load(OUT/(c["regime"]+"_predictions.npz"))[c["method"]]
  elif c["type"]=="tabpfn":
   p=np.zeros_like(y);p[idx["test"]]=prior[c["features"]]
   if c["finish_gate"]:p[finish]=1
  else:p=sum(item["weight"]*get(item["candidate"]) for item in c["components"])
  selectedpred[key]=p
  return p
 report={}
 for key in candidates:report[key]=part_metrics(rows,y,get(key),idx["test"],pairs)
 for label,path in [("frozen_v1_70",ROOT.parent/"scorer_v1/runs/frozen100_20261105/model.json"),("frozen_fewshot_8",ROOT.parent/"fewshot_balanced_v1/results/model.json")]:
  model=json.loads(path.read_text());p=predict_export(model["artifact"],states)
  selectedpred[label]=p;report[label]=part_metrics(rows,y,p,idx["test"],pairs)
 v=json.loads((OUT/"validation.json").read_text());fewshot=[]
 for seed in [101,202,303,404,505]:
  reg=f"fewshot_{seed}";method=v[reg]["selection"];p=np.load(OUT/(reg+"_predictions.npz"))[method]
  entry=part_metrics(rows,y,p,idx["test"],pairs);entry["seed"]=seed;fewshot.append(entry)
 # Report stage-one raw models separately to expose the effect of the generic rule gate.
 stage1=json.loads((OUT/"selection_stage1.json").read_text())
 for reg,method in [("merged_balanced",stage1["selected_method"]),("merged_unweighted","extra_depthNone_leaf2")]:
  p=np.load(OUT/(reg+"_predictions.npz"))[method];key="raw/"+reg+"/"+method
  selectedpred[key]=p;report[key]=part_metrics(rows,y,p,idx["test"],pairs)
 finalkey=selection["selected"];p=get(finalkey)
 prediction_rows=[{"id":rows[i]["id"],"truth":y[i].tolist(),"prediction":p[i].tolist(),"engineered_control":rows[i]["provenance"]["source"]=="synthetic_exact_count_control"} for i in idx["test"]]
 result={"selection_file_sha256":hashlib.sha256((OUT/"selection_final.json").read_bytes()).hexdigest(),"selected":finalkey,"test_n":len(idx["test"]),"comparisons":report,"fewshot_episodes":fewshot,"fewshot_mean":{key:float(np.mean([e[key] for e in fewshot])) for key in ["macro_grade_mae","overall_exact_accuracy"]},"fewshot_overall_mae_mean":float(np.mean([e["mae"]["overall"] for e in fewshot])),"fewshot_overall_mae_std":float(np.std([e["mae"]["overall"] for e in fewshot],ddof=1)),"limits":"17 new-batch development test rows; two grade4, four engineered controls; weak labels, no human calibration"}
 write(OUT/"test_report.json",result);write(OUT/"predictions_test_final.json",prediction_rows)
 with gzip.open(OUT/"candidate_models.json.gz","rt") as f:arts=json.load(f)
 def spec(key):
  c=candidates[key]
  if c["type"]=="builtin":return {"kind":"builtin","artifact":arts[c["regime"]][c["method"]]}
  if c["type"]=="tabpfn":return {"kind":"tabpfn_external","features":c["features"],"finish_gate":c["finish_gate"],"checkpoint_sha256":json.loads((ROOT.parent/"scorer_v2_prior_trial/validation.json").read_text())["checkpoint_sha256"]}
  return {"kind":"weighted_ensemble","members":[{"weight":x["weight"],"model":spec(x["candidate"])} for x in c["components"]]}
 write(OUT/"model.json",{"selected_name":finalkey,"spec":spec(finalkey),"supported_grades":[1,2,3,4],"unsupported_grades":[5],"human_calibrated":False,"ready_for_generation_reward":False})
 for key in ["frozen_v1_70","frozen_fewshot_8",finalkey]:
  m=report[key];print(key,"MAE",m["mae"]["overall"],"MACRO",m["macro_grade_mae"],"EXACT",m["overall_exact_accuracy"],"PAIR",m["matched_pair_order_accuracy"],"NATURAL",m["without_engineered_controls"]["mae"]["overall"])
 print("FEWSHOT",result["fewshot_overall_mae_mean"],result["fewshot_overall_mae_std"])
 print("GRADE4",[(r["id"],r["truth"],r["prediction"]) for r in prediction_rows if r["truth"][0]==4])
if __name__=="__main__":main()
