"""Class-balanced 2-shot development comparison; no foundation model downloads."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import sys,json,hashlib,copy,time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
V1=Path(__file__).resolve().parents[1]/"scorer_v1"
sys.path.insert(0,str(V1))
from scorer import encode_state,card_features,OrdinalMLP,predict_export
from train import metrics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
HASH="85b3d83ca0ed203801161299c83178f858b8e0d3e3a6a288341ca759ada0b59c"
KEYS=("overall","reasoning_depth","move_concealment")
SEEDS=(101,202,303,404,505)
def support_ids(rows,train_ids,k,seed):
 if k<1:raise ValueError("positive k required")
 permitted=set(train_ids);rng=np.random.default_rng(seed);out=[]
 for grade in range(1,5):
  candidates=sorted(r["id"] for r in rows if r["id"] in permitted and r["target"]["overall"]==grade)
  if len(candidates)<k:raise ValueError("insufficient distinct support examples")
  out.extend(rng.choice(candidates,k,replace=False).tolist())
 return sorted(out)
def prototype_fit(x,y,temperature):
 if temperature<=0:raise ValueError("positive temperature required")
 scaler=StandardScaler().fit(x)
 z=scaler.transform(x)
 centres=[];targets=[]
 for g in range(1,5):
  mask=y[:,0]==g
  if not mask.any():raise ValueError("missing support class")
  centres.append(z[mask].mean(0).tolist());targets.append(y[mask].mean(0).tolist())
 return {"kind":"prototype","mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"centres":centres,"targets":targets,"temperature":temperature}
def prototype_predict(model,x):
 z=(x-np.asarray(model["mean"]))/np.asarray(model["scale"])
 dist=((z[:,None,:]-np.asarray(model["centres"])[None,:,:])**2).mean(-1)
 logp=-dist/model["temperature"];logp-=logp.max(1,keepdims=True)
 probs=np.exp(logp);probs/=probs.sum(1,keepdims=True)
 return probs@np.asarray(model["targets"])
def prediction(model,states):
 if model["kind"]=="prototype":return prototype_predict(model,np.stack([encode_state(s) for s in states]))
 return predict_export(model,states)
def score(y,p):
 out=metrics(y,p)
 out["macro_grade_mae"]=float(np.mean(list(out["per_grade_overall_mae"].values())))
 return out
def main():
 if OUT.exists():raise SystemExit("refusing overwrite")
 OUT.mkdir()
 started=time.perf_counter()
 data=ROOT.parent/"pilot100/dataset.jsonl"
 assert hashlib.sha256(data.read_bytes()).hexdigest()==HASH
 rows=[json.loads(s) for s in data.read_text().splitlines()]
 ids=[r["id"] for r in rows]
 split=json.loads((V1/"runs/frozen100_20261105/split.json").read_text())["ids"]
 val=np.array([ids.index(i) for i in split["val"]]);test=np.array([ids.index(i) for i in split["test"]])
 x=np.stack([encode_state(r["input"]) for r in rows]);states=[r["input"] for r in rows]
 y=np.array([[r["target"][k] for k in KEYS] for r in rows],dtype=np.float32)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 episodes=[];exports={};preds={}
 for seed in SEEDS:
  support=support_ids(rows,split["train"],2,seed);ii=np.array([ids.index(i) for i in support])
  registry={};est={}
  registry["support_mean"]={"kind":"constant","mean":y[ii].mean(0).tolist()}
  est["support_mean"]=np.tile(y[ii].mean(0),(len(rows),1))
  for mode in ("counts","board"):
   xx=card_features(x) if mode=="counts" else x;sc=StandardScaler().fit(xx[ii])
   for alpha in (1.,10.,100.):
    name=f"ridge_{mode}_{alpha:g}"
    model=Ridge(alpha=alpha,solver="svd").fit(sc.transform(xx[ii]),y[ii])
    registry[name]={"kind":"ridge","features":mode,"alpha":alpha,"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"coef":model.coef_.tolist(),"intercept":model.intercept_.tolist()}
    est[name]=np.clip(model.predict(sc.transform(xx)),1,4)
  for temp in (.25,1.):
   name=f"prototype_{temp:g}";registry[name]=prototype_fit(x[ii],y[ii],temp)
   est[name]=prototype_predict(registry[name],x)
  sc=StandardScaler().fit(x[ii]);z=sc.transform(x).astype(np.float32)
  torch.manual_seed(1000+seed)
  model=OrdinalMLP();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
  tx=torch.tensor(z[ii]);ty=torch.tensor((y[ii,:,None]>np.arange(1,4)[None,None,:]).astype(np.float32))
  best=float("inf");best_state=None;history=[];best_epoch=0
  for epoch in range(1,301):
   model.train();optimizer.zero_grad()
   loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx),ty);loss.backward();optimizer.step()
   model.eval()
   with torch.no_grad():pv=model.scores(torch.tensor(z[val])).numpy()
   mm=score(y[val],pv)["macro_grade_mae"]
   history.append({"epoch":epoch,"support_bce":float(loss.item()),"val_macro_grade_mae":mm})
   if mm<best-1e-6:best=mm;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
   if epoch-best_epoch>=40:break
  model.load_state_dict(best_state);model.eval()
  with torch.no_grad():est["ordinal_mlp"]=model.scores(torch.tensor(z)).numpy()
  registry["ordinal_mlp"]={"kind":"ordinal_ensemble","mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"weights":[{k:v.tolist() for k,v in best_state.items()}]}
  for name,artifact in registry.items():assert np.allclose(prediction(artifact,states),est[name],atol=2e-5),name
  episodes.append({"seed":seed,"support_ids":support,"support_counts":{str(g):2 for g in range(1,5)},"validation":{name:score(y[val],p[val]) for name,p in est.items()},"mlp_best_epoch":best_epoch,"mlp_history":history})
  exports[str(seed)]=registry;preds[seed]=est
 names=list(preds[SEEDS[0]])
 avgval={name:float(np.mean([e["validation"][name]["macro_grade_mae"] for e in episodes])) for name in names}
 selected=min(names,key=lambda name:(avgval[name],name))
 choice={"method":selected,"criterion":"mean of five validation macro-grade overall MAEs","validation_scores":avgval,"deployed_support_seed":101,"cross_episode_prediction_ensemble":False,"test_used_to_choose":False}
 (OUT/"selection.json").write_text(json.dumps(choice,indent=2),encoding="utf-8")
 # The method is frozen before any test metric below is read.
 for episode in episodes:
  seed=episode["seed"]
  episode["test"]={name:score(y[test],p[test]) for name,p in preds[seed].items()}
 aggregate={}
 for name in names:
  aggregate[name]={}
  for part in ("validation","test"):
   aggregate[name][part]={}
   for metric in ("macro_grade_mae","overall_exact_accuracy"):
    values=[e[part][name][metric] for e in episodes]
    aggregate[name][part][metric]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1))}
   values=[e[part][name]["mae"]["overall"] for e in episodes]
   aggregate[name][part]["overall_mae"]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1))}
  aggregate[name]["test_per_grade_mae"]={str(g):float(np.mean([e["test"][name]["per_grade_overall_mae"][str(g)] for e in episodes])) for g in range(1,5)}
 report={"data_sha256":HASH,"balanced_classes":[1,2,3,4],"k_shot":2,"support_n":8,"support_seeds":list(SEEDS),"val_ids":split["val"],"test_ids":split["test"],"test_status":"previously exposed development comparison, not independent final test","grade4_independent_families_in_support":1,"selected_method":selected,"aggregate":aggregate,"episodes":episodes,"elapsed_seconds":time.perf_counter()-started}
 (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 (OUT/"models.json").write_text(json.dumps(exports),encoding="utf-8")
 modelout={"selected_name":selected,"artifact":exports["101"][selected],"support_ids":episodes[0]["support_ids"],"supported_grades":[1,2,3,4],"unsupported_grades":[5],"human_calibrated":False,"ready_for_generation_reward":False,"status":"balanced_2_shot_development"}
 (OUT/"model.json").write_text(json.dumps(modelout),encoding="utf-8")
 predrows=[{"seed":seed,"id":ids[i],"split":"val" if i in val else "test","truth":y[i].tolist(),"prediction":preds[seed][selected][i].tolist()} for seed in SEEDS for i in list(val)+list(test)]
 (OUT/"predictions.json").write_text(json.dumps(predrows,indent=2),encoding="utf-8")
 print("SELECTED",selected)
 for name,a in aggregate.items():print(name,"val_macro",a["validation"]["macro_grade_mae"],"test_mae",a["test"]["overall_mae"],"test_macro",a["test"]["macro_grade_mae"],"grade4_mae",a["test_per_grade_mae"]["4"])
 print("SECONDS",report["elapsed_seconds"])
if __name__=="__main__":main()
