"""Frozen-100 CPU pilot. Select on validation only; evaluate test after selection."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import json,hashlib,platform,time,copy
from pathlib import Path
import numpy as np
import torch, sklearn, scipy
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scorer import encode_state,split_rows,OrdinalMLP,card_features,predict_export

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/"pilot100/dataset.jsonl"
HASH="85b3d83ca0ed203801161299c83178f858b8e0d3e3a6a288341ca759ada0b59c"
OUT=ROOT/"runs/frozen100_20261105"
KEYS=("overall","reasoning_depth","move_concealment")
def metrics(y,p):
 mae=np.mean(np.abs(y-p),axis=0)
 rho=float(spearmanr(y[:,0],p[:,0]).statistic) if len(np.unique(p[:,0]))>1 else None
 discrete=np.floor(p[:,0]+.5).astype(int)
 pairs=[(i,j) for i in range(len(y)) for j in range(i) if y[i,0]!=y[j,0]]
 rank=np.mean([float((p[i,0]-p[j,0])*(y[i,0]-y[j,0])>0)+.5*float(p[i,0]==p[j,0]) for i,j in pairs]) if pairs else None
 return {"mae":dict(zip(KEYS,map(float,mae))),"mean_mae":float(mae.mean()),
 "overall_spearman":rho,"overall_exact_accuracy":float(np.mean(discrete==y[:,0])),
 "overall_within_one":float(np.mean(np.abs(discrete-y[:,0])<=1)),
 "pairwise_order_accuracy":None if rank is None else float(rank),
 "per_grade_overall_mae":{str(int(g)):float(np.mean(np.abs(y[y[:,0]==g,0]-p[y[:,0]==g,0]))) for g in np.unique(y[:,0])},
 "confusion":[[int(np.sum((y[:,0]==a)&(discrete==b))) for b in range(1,5)] for a in range(1,5)]}
def main():
 if OUT.exists():raise SystemExit("Refusing to overwrite frozen run")
 assert hashlib.sha256(DATA.read_bytes()).hexdigest()==HASH
 OUT.mkdir(parents=True)
 started=time.perf_counter()
 rows=[json.loads(s) for s in DATA.read_text().splitlines()]
 ids=[r["id"] for r in rows];parts=split_rows(rows,20261105)
 idx={k:np.array([ids.index(i) for i in v]) for k,v in parts.items()}
 x=np.stack([encode_state(r["input"]) for r in rows])
 y=np.array([[r["target"][k] for k in KEYS] for r in rows],dtype=np.float32)
 tr,va,te=idx["train"],idx["val"],idx["test"]
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 registry={};preds={};training={}
 registry["train_mean"]={"kind":"constant","mean":y[tr].mean(0).tolist()}
 preds["train_mean"]=np.tile(y[tr].mean(0),(len(rows),1))
 for mode in ("counts","board"):
  z=card_features(x) if mode=="counts" else x
  scaler=StandardScaler().fit(z[tr])
  assert np.allclose(scaler.mean_,z[tr].mean(0),atol=1e-6)
  for alpha in (1.,10.,100.):
   key=f"ridge_{mode}_{alpha:g}"
   model=Ridge(alpha=alpha,solver="svd").fit(scaler.transform(z[tr]),y[tr])
   p=np.clip(model.predict(scaler.transform(z)),1,4)
   registry[key]={"kind":"ridge","features":mode,"alpha":alpha,"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"coef":model.coef_.tolist(),"intercept":model.intercept_.tolist()}
   preds[key]=p
 scaler=StandardScaler().fit(x[tr])
 z=scaler.transform(x).astype(np.float32)
 xt=torch.from_numpy(z[tr]);yt=torch.from_numpy((y[tr,:,None]>np.arange(1,4)[None,None,:]).astype(np.float32))
 xv=torch.from_numpy(z[va]);allx=torch.from_numpy(z)
 weights=[];seedpreds=[]
 for seed in (17,41,73):
  torch.manual_seed(seed);np.random.seed(seed)
  model=OrdinalMLP();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
  best=float("inf");best_state=None;best_epoch=0;history=[]
  for epoch in range(1,301):
   model.train();opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xt),yt)
   loss.backward();opt.step();model.eval()
   with torch.no_grad():p=model.scores(xv).numpy()
   score=float(np.abs(p[:,0]-y[va,0]).mean())
   history.append({"epoch":epoch,"train_bce":float(loss.item()),"val_overall_mae":score})
   if score<best-1e-6:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
   if epoch-best_epoch>=40:break
  model.load_state_dict(best_state);model.eval()
  with torch.no_grad():seedpreds.append(model.scores(allx).numpy())
  weights.append({k:v.tolist() for k,v in best_state.items()})
  training[str(seed)]={"best_epoch":best_epoch,"stopped_epoch":epoch,"best_val_overall_mae":best,"history":history}
 registry["ordinal_mlp_ensemble"]={"kind":"ordinal_ensemble","mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"weights":weights}
 preds["ordinal_mlp_ensemble"]=np.clip(np.mean(seedpreds,axis=0),1,4)
 validation={name:metrics(y[va],p[va]) for name,p in preds.items()}
 selected=min(preds,key=lambda n:(validation[n]["mae"]["overall"],validation[n]["mean_mae"],n))
 # Choice is now fixed; no test information has participated in fitting/selection.
 choice={"name":selected,"rule":"minimum validation overall MAE; tie mean three-head MAE then name","test_not_used_for_selection":True}
 (OUT/"selection.json").write_text(json.dumps(choice,indent=2),encoding="utf-8")
 test={name:metrics(y[te],p[te]) for name,p in preds.items()}
 for name,artifact in registry.items():
  restored=predict_export(artifact,[r["input"] for r in rows])
  assert np.allclose(restored,preds[name],atol=2e-5),name
 modelout={"selected_name":selected,"artifact":registry[selected],"targets":list(KEYS),"supported_grades":[1,2,3,4],"unsupported_grades":[5],"human_calibrated":False,"input_feature_dimension":86,"ready_for_generation_reward":False,"status":"experimental_pilot"}
 (OUT/"model.json").write_text(json.dumps(modelout,ensure_ascii=False),encoding="utf-8")
 (OUT/"models.json").write_text(json.dumps(registry),encoding="utf-8")
 (OUT/"split.json").write_text(json.dumps({"seed":20261105,"data_sha256":HASH,"ids":parts,"distributions":{k:{str(g):sum(int(y[i,0])==g for i in v) for g in range(1,5)} for k,v in idx.items()}},indent=2),encoding="utf-8")
 (OUT/"training.json").write_text(json.dumps(training),encoding="utf-8")
 report={"data_sha256":HASH,"selected":selected,"validation":validation,"test":test,
 "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"scipy":scipy.__version__,"device":"cpu","threads":1},
 "seeds":[17,41,73],"max_epochs":300,"patience":40,"loss":"unweighted ordinal BCE for three targets","feature_scaler_fit":"train only","new_highscore_candidates_used":False,
 "limit":"single-assistant labels; source-held-out small development test, one grade-4 per test split; no grade-5 examples",
 "elapsed_seconds":time.perf_counter()-started}
 (OUT/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 prediction_rows=[]
 for split,ii in idx.items():
  for i in ii:prediction_rows.append({"id":ids[i],"split":split,"truth":y[i].tolist(),"prediction":preds[selected][i].tolist()})
 (OUT/"predictions.json").write_text(json.dumps(prediction_rows,indent=2),encoding="utf-8")
 print(json.dumps({"selected":selected,"split":{k:len(v) for k,v in idx.items()},"val":validation[selected],"test":test[selected],"seed_best_epochs":{s:v["best_epoch"] for s,v in training.items()},"seconds":report["elapsed_seconds"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
