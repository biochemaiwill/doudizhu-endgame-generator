"""Controlled data-volume comparison; no new hyperparameter selection."""
import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np,torch,joblib,sklearn
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/"scorer_v6"))
spec=importlib.util.spec_from_file_location("frozen_v6_train",B/"scorer_v6/train.py")
v6=importlib.util.module_from_spec(spec);spec.loader.exec_module(v6)
from model import features,metric,predict_artifact
from data import components
CONFIGS={
 "rf_balanced":dict(kind="forest",mode="base",balanced=True,depth=None,leaf=1,max_features="sqrt"),
 "mlp_balanced":dict(kind="mlp",mode="relational",balanced=True,lr=.001),
 "cross_balanced":dict(kind="cross",mode="relational",balanced=True,lr=.001),
}
def load():
 p=B/"data300_v1/dataset_merged.jsonl"
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 old=json.loads((B/"scorer_v6/dataset.json").read_text());oldids={r["id"] for r in old}
 lookup={r["id"]:i for i,r in enumerate(rows)}
 assert len(lookup)==len(rows)
 assert len({r["verification"]["key"] for r in rows})==len(rows)
 for r in old:assert rows[lookup[r["id"]]]["input"]==r["input"] and rows[lookup[r["id"]]]["target"]==r["target"]
 edges=[]
 for path in [B/"expansion_v2/pairs.json",B/"expansion_v3/pairs.json",B/"mechanism_pairs_v1/pairs.json",B/"mechanism_pairs_v2/pairs.json",B/"data300_v1/pairs.json"]:
  if path.exists():
   for p in json.loads(path.read_text()):
    if p.get("hard_id") in lookup and p.get("easy_id") in lookup:edges.append({"hard_id":p["hard_id"],"easy_id":p["easy_id"]})
 origin={r["provenance"].get("origin_id"):r["id"] for r in rows}
 for r in rows:
  for name in ["parent_id","parent_origin","augmentation_parent"]:
   parent=r["provenance"].get(name)
   if parent:
    parent=parent if parent in lookup else origin.get(parent)
    if parent in lookup:edges.append(dict(hard_id=parent,easy_id=r["id"]))
 groups=components(rows,edges);owner={i:g for g,ids in groups.items() for i in ids}
 source=json.loads((B/"scorer_v6/cv_split.json").read_text())
 folds=[]
 newids=set(lookup)-oldids
 for f in source["folds"]:
  val=set(f["val_ids"]);oldtrain=set(f["train_ids"])
  forbidden={owner[i] for i in val}
  assert not {owner[i] for i in oldtrain}&forbidden,"new family bridge changes old split"
  added=sorted(i for i in newids if owner[i] not in forbidden)
  excluded=sorted(newids-set(added))
  train=list(f["train_ids"])+added
  assert not {owner[i] for i in train}&forbidden
  folds.append(dict(seed=f["seed"],fold=f["fold"],val_ids=f["val_ids"],old_train_ids=f["train_ids"],expanded_train_ids=train,added_ids=added,excluded_new_related_ids=excluded))
 return rows,lookup,folds,groups
def save(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def main():
 if (R/"results.json").exists():raise SystemExit("refuse overwrite")
 dataset_path=B/"data300_v1/dataset_merged.jsonl"
 digest=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
 rows,lookup,folds,groups=load();torch.set_num_threads(1);start=time.time()
 y=np.array([[r["target"][k]-1 for k in ["overall","reasoning_depth","move_concealment"]] for r in rows])
 states=[r["input"] for r in rows];x={m:np.stack([features(s,m) for s in states]) for m in ["base","relational"]}
 save("split_audit.json",dict(folds=folds,groups=groups,new_n=len(rows)-189,total_n=len(rows),same_old_validation_ids=True))
 results={};predictions={}
 for name,cfg in CONFIGS.items():
  results[name]={};predictions[name]={}
  for arm in ["old","expanded"]:
   fs=[];ps=[]
   for f in folds:
    tr=np.array([lookup[i] for i in f[arm+"_train_ids"]]);va=np.array([lookup[i] for i in f["val_ids"]])
    artifact=v6.fit(cfg,x[cfg["mode"]],y,tr,f["seed"]*100+f["fold"])
    pred=predict_artifact(artifact,[states[i] for i in va])
    fs.append(dict(seed=f["seed"],fold=f["fold"],train_n=len(tr),**metric(y[va],pred)))
    ps += [dict(id=rows[i]["id"],seed=f["seed"],fold=f["fold"],truth=y[i].tolist(),probabilities=pred[j].tolist()) for j,i in enumerate(va)]
    print(name,arm,f["seed"],f["fold"],round(fs[-1]["accuracy"],4),flush=True)
   results[name][arm]=dict(config=cfg,folds=fs,mean_accuracy=float(np.mean([f["accuracy"] for f in fs])),mean_macro_recall=float(np.mean([f["macro_recall"] for f in fs])),mean_recall=np.mean([f["recall"] for f in fs],axis=0).tolist())
   predictions[name][arm]=ps
   save("progress.json",results)
  results[name]["accuracy_delta"]=results[name]["expanded"]["mean_accuracy"]-results[name]["old"]["mean_accuracy"]
 assert hashlib.sha256(dataset_path.read_bytes()).hexdigest()==digest,"dataset changed during training"
 save("results.json",results);save("oof_predictions.json",predictions)
 winner=max(CONFIGS,key=lambda k:(results[k]["expanded"]["mean_accuracy"],results[k]["expanded"]["mean_macro_recall"],k))
 save("selection.json",dict(name=winner,criterion="expanded development CV accuracy then macro recall; fixed three-config comparison",strict_test_evaluated=False))
 model=v6.fit(CONFIGS[winner],x[CONFIGS[winner]["mode"]],y,np.arange(len(rows)),917)
 joblib.dump(model,R/"model.joblib.gz",compress=3)
 restored=joblib.load(R/"model.joblib.gz")
 assert np.allclose(predict_artifact(model,states),predict_artifact(restored,states))
 save("environment.json",dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,threads=1,epochs=v6.EPOCHS,cv_seeds=v6.SEEDS,final_seed=917,seconds=time.time()-start,dataset_sha256=hashlib.sha256((B/"data300_v1/dataset_merged.jsonl").read_bytes()).hexdigest()))
 print("DONE",winner,{k:{"old":v["old"]["mean_accuracy"],"expanded":v["expanded"]["mean_accuracy"],"delta":v["accuracy_delta"]} for k,v in results.items()},flush=True)
if __name__=="__main__":main()
