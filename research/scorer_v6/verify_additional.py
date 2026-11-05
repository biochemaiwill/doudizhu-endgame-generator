import json,joblib
from pathlib import Path
import numpy as np
from model import features
R=Path(__file__).resolve().parent
r=json.loads((R/"additional_cv.json").read_text());folds=json.loads((R/"cv_split.json").read_text())["folds"]
for name,row in r.items():
 records=row["oof_predictions"];acc=[]
 for i,f in enumerate(folds):
  part=[p for p in records if p["seed"]==f["seed"] and p["fold"]==f["fold"]]
  assert {p["id"] for p in part}==set(f["val_ids"])
  a=np.mean([p["truth"]==p["prediction"] for p in part]);acc.append(a)
  assert abs(a-row["folds"][i]["accuracy"])<1e-12
 assert abs(np.mean(acc)-row["accuracy"])<1e-12
selection=json.loads((R/"additional_selection.json").read_text())
expected=max([n for n in r if r[n]["config"]["kind"]=="hgb"],key=lambda n:(r[n]["accuracy"],r[n]["macro_recall"],n))
assert selection["selected"]==expected and not selection["strict_holdout_read"]
m=joblib.load(R/"additional_model.joblib.gz");assert m["config"]["kind"]=="hgb"
rows=json.loads((R/"dataset.json").read_text())
x=np.stack([features(p["input"],"base") for p in rows[:5]])
a=m["model"].predict(x);assert set(a)<=set([0,1,2,3])
(R/"additional_verification.json").write_text(json.dumps({"candidate_fold_metrics_checked":len(r)*len(folds),"CE_eligible_selection_matches":True,"artifact_HGB_predictions_valid":True},indent=2))
print("PASS",len(r)*len(folds),"candidate-fold metrics and HGB-only selection")
