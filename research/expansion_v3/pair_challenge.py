"""Frozen scorer stress test: same counts, same card multiset, no one-move finishes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"research/scorer_v2"))
from predict import predict_spec
def main():
 if (ROOT/"pair_challenge.json").exists():raise SystemExit("refuse overwrite challenge")
 rows={}
 for path in [BASE/"research/scorer_v2/dataset.jsonl",ROOT/"dataset.jsonl"]:
  rows.update({r["id"]:r for r in map(json.loads,path.read_text().splitlines())})
 pairs=json.loads((ROOT/"pairs.json").read_text())
 ids=sorted({p[k] for p in pairs for k in ["hard_id","easy_id"]})
 model=json.loads((BASE/"research/scorer_v2/run/model.json").read_text())
 pred=predict_spec(model["spec"],[rows[i]["input"] for i in ids]);lookup=dict(zip(ids,pred.tolist()))
 result=[]
 for p in pairs:
  hard,easy=p["hard_id"],p["easy_id"];gap=lookup[hard][0]-lookup[easy][0]
  result.append(dict(pair_id=p["id"],hard_id=hard,easy_id=easy,hard_label=rows[hard]["target"]["overall"],easy_label=rows[easy]["target"]["overall"],hard_prediction=lookup[hard],easy_prediction=lookup[easy],predicted_gap=gap,correct_order=gap>0))
 output={"model":model["selected_name"],"model_spec_sha256":hashlib.sha256((BASE/"research/scorer_v2/run/model.json").read_bytes()).hexdigest(),"n_pairs":len(result),"correct":sum(p["correct_order"] for p in result),"pair_order_accuracy":float(np.mean([p["correct_order"] for p in result])),"mean_predicted_gap":float(np.mean([p["predicted_gap"] for p in result])),"pairs":result,"status":"counterfactual diagnostic on related families, not independent final test; no fitting or model change"}
 (ROOT/"pair_challenge.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({k:v for k,v in output.items() if k!="pairs"},ensure_ascii=False))
if __name__=="__main__":main()
