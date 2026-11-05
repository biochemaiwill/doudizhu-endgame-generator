"""Development-only grouped CV. Never reads strict_holdout_v6."""
import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys,json,time,hashlib,platform,copy
from pathlib import Path
import numpy as np,torch,sklearn,joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from model import features,CENet,metric,predict_artifact
from data import components
R=Path(__file__).resolve().parent;B=R.parent
SEEDS=[17,73];EPOCHS=100
def save(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def make_data():
 rows=[]
 for p in [B/"scorer_v4/dataset.jsonl",B/"grade4_expansion_v5/dataset.jsonl"]:rows += [json.loads(x) for x in p.read_text().splitlines()]
 assert len(rows)==189 and len({r["verification"]["key"] for r in rows})==189
 pairs=json.loads((B/"expansion_v2/pairs.json").read_text())+json.loads((B/"expansion_v3/pairs.json").read_text())
 for r in rows:
  parent=r["provenance"].get("parent_id")
  if parent:pairs.append({"hard_id":parent,"easy_id":r["id"]})
 groups=components(rows,pairs);owner={i:g for g,ids in groups.items() for i in ids}
 anchor={g for g,ids in groups.items() if any(next(r for r in rows if r["id"]==i)["provenance"]["source"]=="MasterMind train API" for i in ids)}
 eligible=np.array([i for i,r in enumerate(rows) if owner[r["id"]] not in anchor])
 anchored=np.array([i for i,r in enumerate(rows) if owner[r["id"]] in anchor])
 y=np.array([[r["target"][k]-1 for k in ["overall","reasoning_depth","move_concealment"]] for r in rows])
 folds=[]
 for seed in SEEDS:
  cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
  for fold,(it,iv) in enumerate(cv.split(eligible,y[eligible,0],[owner[rows[i]["id"]] for i in eligible])):
   tr=np.r_[eligible[it],anchored];va=eligible[iv]
   assert not set(owner[rows[i]["id"]] for i in tr)&set(owner[rows[i]["id"]] for i in va)
   assert set(y[tr,0])=={0,1,2,3}
   folds.append(dict(seed=seed,fold=fold,tr=tr,va=va))
 return rows,y,folds,groups
def fit(cfg,x,y,tr,seed):
 kind=cfg["kind"];mode=cfg["mode"];balanced=cfg["balanced"]
 if kind in ["extra","forest"]:
  cls=ExtraTreesClassifier if kind=="extra" else RandomForestClassifier
  estimators=[]
  for h in range(3):
   m=cls(n_estimators=192,max_depth=cfg["depth"],min_samples_leaf=cfg["leaf"],max_features=cfg["max_features"],criterion="log_loss",class_weight="balanced" if balanced else None,n_jobs=1,random_state=seed)
   m.fit(x[tr],y[tr,h]);estimators.append(m)
  return {"kind":"tree","mode":mode,"estimators":estimators}
 torch.manual_seed(seed)
 start=45 if kind=="cross" else 0
 sc=StandardScaler().fit(x[tr,start:]);z=x.copy();z[:,start:]=sc.transform(x[:,start:])
 m=CENet(kind,x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=cfg["lr"],weight_decay=.01)
 tx=torch.tensor(z[tr],dtype=torch.float32);ty=torch.tensor(y[tr],dtype=torch.long)
 weights=[]
 for h in range(3):
  n=np.bincount(y[tr,h],minlength=4);weights.append(torch.tensor(len(tr)/(4*np.maximum(n,1)),dtype=torch.float32) if balanced else None)
 for ep in range(EPOCHS):
  m.train();opt.zero_grad();logits=m(tx)
  loss=sum(w*torch.nn.functional.cross_entropy(logits[:,h],ty[:,h],weight=weights[h]) for h,w in enumerate([1.,.25,.25]))
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 return {"kind":"neural","net":kind,"mode":mode,"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"weights":{k:v.tolist() for k,v in m.state_dict().items()}}
def main():
 if (R/"selection.json").exists():raise SystemExit("selection frozen")
 torch.set_num_threads(1);start=time.time()
 rows,y,folds,groups=make_data();states=[r["input"] for r in rows]
 save("dataset.json",rows)
 save("cv_split.json",{"anchor":"MasterMind source group always train","groups":groups,"folds":[dict(seed=f["seed"],fold=f["fold"],train_ids=[rows[i]["id"] for i in f["tr"]],val_ids=[rows[i]["id"] for i in f["va"]],val_grade_counts=np.bincount(y[f["va"],0],minlength=4).tolist()) for f in folds]})
 configs=[]
 for mode in ["base","relational"]:
  for kind in ["extra","forest"]:
   for balanced in [False,True]:
    for leaf in [1,3]:
     configs.append(dict(kind=kind,mode=mode,balanced=balanced,depth=None,leaf=leaf,max_features=1. if kind=="extra" else "sqrt"))
 for kind in ["mlp","cross"]:
  for balanced in [False,True]:
   configs.append(dict(kind=kind,mode="relational",balanced=balanced,lr=.001))
 config_name=lambda c:"_".join(f"{k}-{v}" for k,v in c.items())
 cache={mode:np.stack([features(s,mode) for s in states]) for mode in ["base","relational"]}
 results={};all_preds={}
 for cfg in configs:
  name=config_name(cfg);foldstats=[];records=[]
  for f in folds:
   a=fit(cfg,cache[cfg["mode"]],y,f["tr"],f["seed"]*100+f["fold"])
   p=predict_artifact(a,[states[i] for i in f["va"]])
   met=metric(y[f["va"]],p);foldstats.append(met)
   records += [dict(id=rows[i]["id"],seed=f["seed"],truth=y[i].tolist(),probabilities=p[j].tolist()) for j,i in enumerate(f["va"])]
  results[name]={"config":cfg,"mean_accuracy":float(np.mean([f["accuracy"] for f in foldstats])),"std_accuracy":float(np.std([f["accuracy"] for f in foldstats],ddof=1)),"mean_macro_recall":float(np.mean([f["macro_recall"] for f in foldstats])),"folds":foldstats}
  all_preds[name]=records;save("cv_results.json",results)
  print(name,round(results[name]["mean_accuracy"],4),round(results[name]["mean_macro_recall"],4),flush=True)
 selected=max(results,key=lambda n:(results[n]["mean_accuracy"],results[n]["mean_macro_recall"],n))
 choice={"name":selected,"config":results[selected]["config"],"criterion":"mean group CV overall accuracy then macro recall","external_test_read":False,"cv_is_development":True,"target_accuracy":.9}
 save("selection.json",choice)
 model=fit(choice["config"],cache[choice["config"]["mode"]],y,np.arange(len(rows)),917)
 joblib.dump(model,R/"model.joblib.gz",compress=3)
 reload=joblib.load(R/"model.joblib.gz");assert np.allclose(predict_artifact(model,states),predict_artifact(reload,states))
 save("oof_predictions.json",all_preds[selected])
 save("environment.json",dict(python=sys.version,torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,epochs=EPOCHS,cv_seeds=SEEDS,final_seed=917,seconds=time.time()-start,device="cpu",threads=1,dataset_sha256=hashlib.sha256((R/"dataset.json").read_bytes()).hexdigest()))
 print("SELECTED",selected,results[selected]["mean_accuracy"],flush=True)
if __name__=="__main__":main()
