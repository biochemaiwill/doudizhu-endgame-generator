import sys,os,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
sys.path[:0]=[str(B/"scorer_v2_prior_trial/runtime"),str(B/"scorer_v6")]
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np,torch
from tabpfn import TabPFNClassifier
from model import features
torch.set_num_threads(1)
rows={r["id"]:r for r in json.loads((B/"scorer_v6/dataset.json").read_text())}; folds=json.loads((B/"scorer_v6/cv_split.json").read_text())["folds"]
xx={i:features(r["input"],mode="base") for i,r in rows.items()}
out={"config":{"name":"TabPFN_v2_classifier_base143","n_estimators":2,"random_state":73,"decision":"classes_[argmax(probabilities)]"},"checkpoint_sha256":hashlib.sha256((D/"classifier.ckpt").read_bytes()).hexdigest(),"strict_holdout_read":False,"folds":[]}
for f in folds:
 tr=f["train_ids"];va=f["val_ids"]
 clf=TabPFNClassifier(model_path=D/"classifier.ckpt",device="cpu",n_estimators=2,n_jobs=1,random_state=73)
 clf.fit(np.stack([xx[i] for i in tr]),np.array([rows[i]["target"]["overall"] for i in tr]))
 p=clf.predict_proba(np.stack([xx[i] for i in va]));pred=clf.classes_[p.argmax(1)];y=np.array([rows[i]["target"]["overall"] for i in va]);rec=[float(np.mean(pred[y==g]==g)) for g in range(1,5)]
 z={**f,"classes":clf.classes_.tolist(),"probabilities":p.tolist(),"predictions":pred.tolist(),"truth":y.tolist(),"accuracy":float(np.mean(pred==y)),"macro_recall":float(np.mean(rec)),"per_grade_recall":rec}
 out["folds"].append(z);(D/"cv_results.json").write_text(json.dumps(out,indent=2));print(f["seed"],f["fold"],z["accuracy"],flush=True)
out["mean_accuracy"]=float(np.mean([f["accuracy"] for f in out["folds"]]));out["mean_macro_recall"]=float(np.mean([f["macro_recall"] for f in out["folds"]]));out["per_grade_recall"]=np.mean([f["per_grade_recall"] for f in out["folds"]],0).tolist()
(D/"cv_results.json").write_text(json.dumps(out,indent=2));print("DONE",out["mean_accuracy"],out["per_grade_recall"],flush=True)
