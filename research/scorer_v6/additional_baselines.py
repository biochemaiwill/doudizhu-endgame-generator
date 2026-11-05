"""Twelve prespecified overall-only classifier baselines on the frozen six folds."""
import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import json,time,sys,hashlib
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from model import features
R=Path(__file__).resolve().parent
def write(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def score(y,p):
 cm=confusion_matrix(y,p,labels=[0,1,2,3]);rec=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(4),where=cm.sum(1)!=0)
 return {"accuracy":float((y==p).mean()),"macro_recall":float(rec.mean()),"recall":rec.tolist(),"confusion":cm.tolist()}
def fit(cfg,x,y,tr):
 scaler=None
 if cfg["kind"]=="hgb":
  m=HistGradientBoostingClassifier(loss="log_loss",max_iter=200,learning_rate=cfg["lr"],max_leaf_nodes=cfg["leaves"],min_samples_leaf=cfg["min_leaf"],random_state=773)
  m.fit(x[tr],y[tr])
 else:
  scaler=StandardScaler().fit(x[tr])
  m=SVC(C=cfg["C"],gamma="scale",kernel="rbf",class_weight=cfg["class_weight"],probability=False)
  m.fit(scaler.transform(x[tr]),y[tr])
 return m,scaler
def main():
 if (R/"additional_selection.json").exists():raise SystemExit("additional selection frozen")
 start=time.time();rows=json.loads((R/"dataset.json").read_text());folds=json.loads((R/"cv_split.json").read_text())["folds"]
 lookup={r["id"]:i for i,r in enumerate(rows)}
 x=np.stack([features(r["input"],"base") for r in rows]);y=np.array([r["target"]["overall"]-1 for r in rows])
 configs=[{"kind":"hgb","lr":lr,"leaves":leaves,"min_leaf":ml} for lr in [.05,.1] for leaves in [7,15] for ml in [5,10]]
 configs += [{"kind":"svc","C":c,"class_weight":w} for c in [1.,10.] for w in [None,"balanced"]]
 write("additional_protocol.json",{"configurations":configs,"features":"base143","fold_file_sha256":hashlib.sha256((R/"cv_split.json").read_bytes()).hexdigest(),"selection":"mean accuracy, then macro recall","target":"overall only; no depth/concealment claims","strict_holdout_read":False})
 out={}
 for cfg in configs:
  name=json.dumps(cfg,sort_keys=True);parts=[];records=[]
  for f in folds:
   tr=np.array([lookup[i] for i in f["train_ids"]]);va=np.array([lookup[i] for i in f["val_ids"]])
   model,sc=fit(cfg,x,y,tr);pred=model.predict(sc.transform(x[va]) if sc is not None else x[va])
   parts.append(score(y[va],pred));records += [{"id":rows[i]["id"],"seed":f["seed"],"fold":f["fold"],"truth":int(y[i])+1,"prediction":int(p)+1} for i,p in zip(va,pred)]
  out[name]={"config":cfg,"accuracy":float(np.mean([p["accuracy"] for p in parts])),"std_accuracy":float(np.std([p["accuracy"] for p in parts],ddof=1)),"macro_recall":float(np.mean([p["macro_recall"] for p in parts])),"folds":parts,"oof_predictions":records}
  write("additional_cv.json",out);print(cfg,out[name]["accuracy"],out[name]["macro_recall"],flush=True)
 diagnostic=max(out,key=lambda n:(out[n]["accuracy"],out[n]["macro_recall"],n))
 winner=max([n for n in out if out[n]["config"]["kind"]=="hgb"],key=lambda n:(out[n]["accuracy"],out[n]["macro_recall"],n))
 write("additional_diagnostic_selection.json",{"selected":diagnostic,"eligible_for_CE_delivery":out[diagnostic]["config"]["kind"]=="hgb"})
 selected=out[winner];m,sc=fit(selected["config"],x,y,np.arange(len(y)))
 a={"kind":"overall_only","model":m,"scaler":sc,"feature_mode":"base","config":selected["config"],"class_labels":[1,2,3,4]}
 joblib.dump(a,R/"additional_model.joblib.gz",compress=3)
 loaded=joblib.load(R/"additional_model.joblib.gz")
 np.testing.assert_array_equal(m.predict(sc.transform(x) if sc is not None else x),loaded["model"].predict(loaded["scaler"].transform(x) if loaded["scaler"] is not None else x))
 write("additional_selection.json",{"selected":winner,"config":selected["config"],"cv_accuracy":selected["accuracy"],"cv_macro_recall":selected["macro_recall"],"strict_holdout_read":False,"model_outputs":"overall only","elapsed_seconds":time.time()-start})
 print("BEST",winner,selected["accuracy"],flush=True)
if __name__=="__main__":main()
