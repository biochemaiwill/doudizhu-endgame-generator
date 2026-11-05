"""Predict 1-4 difficulty from boards; no endgame search is performed."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from experiment import bundle_predict
from models import encode_structured
ROOT=Path(__file__).resolve().parent
def predict_spec(spec,states):
 if spec["kind"]=="builtin":return bundle_predict(spec["artifact"],states)
 if spec["kind"]=="weighted_ensemble":
  return sum(item["weight"]*predict_spec(item["model"],states) for item in spec["members"])
 if spec["kind"]=="tabpfn_external":
  worker=ROOT.parent/"scorer_v2_prior_trial/predict.py"
  code="import runpy,sys,json; m=runpy.run_path(sys.argv[1],run_name='ddz_prior_worker'); p=m['predict'](json.load(sys.stdin),features=sys.argv[2]); print('DDZ_RESULT:'+json.dumps(p.tolist()))"
  done=subprocess.run([sys.executable,"-c",code,str(worker),spec["features"]],input=json.dumps(states),capture_output=True,text=True,timeout=240,check=True)
  lines=[line[len("DDZ_RESULT:"):] for line in done.stdout.splitlines() if line.startswith("DDZ_RESULT:")]
  if len(lines)!=1:raise RuntimeError("prior worker did not return exactly one prediction batch")
  out=np.array(json.loads(lines[0]),dtype=float)
  if spec["finish_gate"]:
   for i,s in enumerate(states):
    if encode_structured(s)[-1]>.5:out[i]=1.
  return out
 raise ValueError("unsupported model spec")
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--model",type=Path,default=ROOT/"run/model.json")
 args=p.parse_args();raw=json.loads(args.input.read_text(encoding="utf-8"));one=isinstance(raw,dict);states=[raw] if one else raw
 model=json.loads(args.model.read_text());values=predict_spec(model["spec"],states)
 if values.shape!=(len(states),3) or not np.isfinite(values).all():raise ValueError("invalid model predictions")
 result=[{"overall":float(v[0]),"reasoning_depth":float(v[1]),"move_concealment":float(v[2]),"supported_grades":[1,2,3,4],"grade_5":"unsupported","human_calibrated":False,"ready_for_generation_reward":False,"model":model["selected_name"]} for v in values]
 print(json.dumps(result[0] if one else result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
