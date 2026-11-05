import sys,os,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
sys.path[:0]=[str(B/"scorer_v2_prior_trial/runtime"),str(B/"scorer_v6")]
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np,torch
from tabpfn import TabPFNClassifier
from model import features
torch.set_num_threads(1)
path=B/"data300_v1/dataset_merged.jsonl";digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest=="c8d107007c9ea6ca3a05416a31b4f56b92996820aa22ef7699377fd75ac133e0"
rows={r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
folds=json.loads((B/"scorer_data300_v1/split_audit.json").read_text())["folds"]
old=json.loads((B/"scorer_v6_tabpfn/cv_results.json").read_text())
out={"config":old["config"],"dataset_sha256":digest,"old_mean_accuracy":old["mean_accuracy"],"old_per_grade_recall":old["per_grade_recall"],"folds":[],"strict_holdout_read":False}
xx={i:features(r["input"],mode="base") for i,r in rows.items()}
for f,o in zip(folds,old["folds"]):
 assert f["val_ids"]==o["val_ids"] and f["old_train_ids"]==o["train_ids"]
 tr=f["expanded_train_ids"];va=f["val_ids"]
 m=TabPFNClassifier(model_path=B/"scorer_v6_tabpfn/classifier.ckpt",device="cpu",n_estimators=2,n_jobs=1,random_state=73)
 m.fit(np.stack([xx[i] for i in tr]),np.array([rows[i]["target"]["overall"] for i in tr]))
 p=m.predict_proba(np.stack([xx[i] for i in va]));pred=m.classes_[p.argmax(1)];y=np.array([rows[i]["target"]["overall"] for i in va])
 rec=[float(np.mean(pred[y==g]==g)) for g in range(1,5)]
 z={**f,"classes":m.classes_.tolist(),"probabilities":p.tolist(),"predictions":pred.tolist(),"truth":y.tolist(),"old_accuracy":o["accuracy"],"accuracy":float(np.mean(pred==y)),"macro_recall":float(np.mean(rec)),"per_grade_recall":rec}
 out["folds"].append(z);(D/"results.json").write_text(json.dumps(out,indent=2));print(f["seed"],f["fold"],z["accuracy"],flush=True)
out["expanded_mean_accuracy"]=float(np.mean([f["accuracy"] for f in out["folds"]]));out["expanded_macro_recall"]=float(np.mean([f["macro_recall"] for f in out["folds"]]));out["expanded_per_grade_recall"]=np.mean([f["per_grade_recall"] for f in out["folds"]],0).tolist();out["accuracy_delta"]=out["expanded_mean_accuracy"]-out["old_mean_accuracy"]
(D/"results.json").write_text(json.dumps(out,indent=2));print("DONE",out["expanded_mean_accuracy"],out["accuracy_delta"],out["expanded_per_grade_recall"])
