"""Frozen grouped development experiment; select on validation before test metrics."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import sys,json,gzip,time,copy,hashlib,platform
from pathlib import Path
import numpy as np
import torch,sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from data import class_weights,support_ids
from models import encode_structured,encode_state,card_features,OrdinalMLP,RoleOrdinalNet,STRUCTURED_DIM,export_torch,predict_artifact
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scorer_v1"))
from train import metrics as base_metrics
from scorer import predict_export
ROOT=Path(__file__).resolve().parent;OUT=ROOT/"run"
TARGETS=("overall","reasoning_depth","move_concealment")
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def bundle_predict(a,states):
 if a["kind"]=="finish_gate":
  result=bundle_predict(a["base"],states).copy()
  for i,state in enumerate(states):
   if encode_structured(state)[-1]>0.5:result[i]=1.
  return result
 if a["kind"]=="ensemble":return np.mean([bundle_predict(x,states) for x in a["members"]],axis=0)
 return predict_artifact(a,states)
def stats(y,p):
 r=base_metrics(y,p);r["macro_grade_mae"]=float(np.mean(list(r["per_grade_overall_mae"].values())))
 return r
def rank_loss(scores,y,w):
 diff=y[:,None]-y[None,:];mask=diff>0
 if not mask.any():return scores.sum()*0
 z=scores[:,None]-scores[None,:]
 pw=w[:,None]*w[None,:]
 return (torch.nn.functional.softplus(.25-z[mask])*pw[mask]).sum()/pw[mask].sum()
def fit_suite(fx,y,tr,va,balanced,seeds,states):
 w=class_weights(y[tr,0]) if balanced else np.ones(len(tr),np.float32)
 artifacts={};pred={};logs={}
 artifacts["constant"]={"kind":"constant","mean":np.average(y[tr],axis=0,weights=w).tolist()}
 pred["constant"]=bundle_predict(artifacts["constant"],states)
 for mode,alphas in [("counts",[1.,10.,100.]),("board",[1.,10.,100.]),("structured",[1.,10.,100.])]:
  xx=fx[mode];sc=StandardScaler().fit(xx[tr])
  for alpha in alphas:
   model=Ridge(alpha=alpha,solver="svd").fit(sc.transform(xx[tr]),y[tr],sample_weight=w)
   key=f"ridge_{mode}_{alpha:g}";a={"kind":"ridge","features":mode,"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"coef":model.coef_.tolist(),"intercept":model.intercept_.tolist(),"alpha":alpha}
   artifacts[key]=a;pred[key]=np.clip(model.predict(sc.transform(xx)),1,4)
 for tree_kind,depth,leaf in [("extra",4,2),("extra",None,2),("extra",None,1),("forest",None,2)]:
  cls=ExtraTreesRegressor if tree_kind=="extra" else RandomForestRegressor
  model=cls(n_estimators=128,max_depth=depth,min_samples_leaf=leaf,random_state=317,n_jobs=1,max_features=1.0)
  model.fit(fx["structured"][tr],y[tr],sample_weight=w)
  key=f"{tree_kind}_depth{depth}_leaf{leaf}"
  a={"kind":"tree_ensemble","features":"structured","trees":[{"children_left":e.tree_.children_left.tolist(),"children_right":e.tree_.children_right.tolist(),"feature":e.tree_.feature.tolist(),"threshold":e.tree_.threshold.tolist(),"value":e.tree_.value.reshape(e.tree_.node_count,3).tolist()} for e in model.estimators_]}
  artifacts[key]=a;pred[key]=np.clip(model.predict(fx["structured"]),1,4)
 for kind,rankcoef in [("ordinal_mlp",0.),("structured_mlp",0.),("role_ordinal",0.),("role_ordinal",.15)]:
  mode="board" if kind=="ordinal_mlp" else "structured"
  xx=fx[mode];sc=StandardScaler().fit(xx[tr]);z=sc.transform(xx).astype(np.float32)
  xt=torch.tensor(z[tr]);tv=torch.tensor(z[va]);allx=torch.tensor(z)
  ty=torch.tensor((y[tr,:,None]>np.arange(1,4)[None,None,:]).astype(np.float32))
  tw=torch.tensor(w);rawy=torch.tensor(y[tr,0])
  members=[];ps=[];records=[]
  for seed in seeds:
   torch.manual_seed(seed)
   model=RoleOrdinalNet() if kind=="role_ordinal" else OrdinalMLP(input_dim=xx.shape[1])
   opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
   best=float("inf");epoch_best=0;best_state=None;history=[]
   for epoch in range(1,401):
    model.train();opt.zero_grad();logits=model(xt)
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction="none").mean((1,2))
    loss=(losses*tw).sum()/tw.sum()
    if rankcoef:loss=loss+rankcoef*rank_loss(1+torch.sigmoid(logits[:,0,:]).sum(-1),rawy,tw)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
    model.eval()
    with torch.no_grad():pv=model.scores(tv).numpy()
    metric=stats(y[va],pv)["macro_grade_mae"]
    if epoch%10==0:history.append([epoch,float(loss.item()),metric])
    if metric<best-1e-6:best=metric;epoch_best=epoch;best_state=copy.deepcopy(model.state_dict())
    if epoch-epoch_best>=50:break
   model.load_state_dict(best_state);model.eval()
   with torch.no_grad():ps.append(model.scores(allx).numpy())
   members.append(export_torch(model,kind,sc));records.append({"seed":seed,"best_epoch":epoch_best,"stopped_epoch":epoch,"val_macro_mae":best,"history_every10":history})
  key=kind+("_rank" if rankcoef else "")
  artifacts[key]={"kind":"ensemble","members":members};pred[key]=np.mean(ps,axis=0);logs[key]=records
 # Portable export must match training-library prediction before selecting a model.
 for key,a in artifacts.items():
  assert np.allclose(bundle_predict(a,states),pred[key],atol=3e-5),key
 return artifacts,pred,logs
def part_metrics(rows,y,pred,idx,pairs):
 result=stats(y[idx],pred[idx]);positions=set(idx.tolist());ids={r["id"]:i for i,r in enumerate(rows)}
 natural=np.array([i for i in idx if rows[i]["provenance"]["source"]!="synthetic_exact_count_control"])
 result["without_engineered_controls"]=stats(y[natural],pred[natural]) if len(natural) else None
 pair_results=[]
 for pair in pairs:
  a,b=ids[pair["hard_id"]],ids[pair["easy_id"]]
  if a in positions and b in positions:
   gap=float(pred[a,0]-pred[b,0])
   pair_results.append({"id":pair["id"],"hard":rows[a]["id"],"easy":rows[b]["id"],"predicted_gap":gap,"true_gap":float(y[a,0]-y[b,0]),"correct_order":gap>0})
 result["matched_pairs"]=pair_results
 result["matched_pair_order_accuracy"]=float(np.mean([p["correct_order"] for p in pair_results])) if pair_results else None
 return result
def main():
 if OUT.exists():raise SystemExit("refuse overwrite")
 OUT.mkdir();start=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rows=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
 split=json.loads((ROOT/"split.json").read_text());idmap={r["id"]:i for i,r in enumerate(rows)}
 idx={p:np.array([idmap[i] for i in ids]) for p,ids in split["ids"].items()}
 states=[r["input"] for r in rows]
 board=np.stack([encode_state(s) for s in states])
 fx={"board":board,"counts":card_features(board),"structured":np.stack([encode_structured(s) for s in states])}
 y=np.array([[r["target"][k] for k in TARGETS] for r in rows],np.float32)
 pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
 old=np.array([i for i,r in enumerate(rows) if r["id"].startswith("DZ")])
 regimes={};exports={};allpred={};fitting={}
 configs=[("old100_balanced",old,True),("merged_balanced",idx["train"],True),("merged_unweighted",idx["train"],False)]
 for name,tr,balanced in configs:
  arts,ps,logs=fit_suite(fx,y,tr,idx["val"],balanced,[17,41,73],states)
  val={k:part_metrics(rows,y,p,idx["val"],pairs) for k,p in ps.items()}
  selected=min(val,key=lambda k:(val[k]["macro_grade_mae"],val[k]["mae"]["overall"],k))
  regimes[name]={"train_ids":[rows[i]["id"] for i in tr],"balanced":balanced,"selection":selected,"validation":val}
  exports[name]=arts;allpred[name]=ps;fitting[name]=logs
  print(name,"SELECTED",selected,"VAL",val[selected]["macro_grade_mae"],flush=True)
 # Each episode has eight actual training examples; no across-episode deployment ensemble.
 for seed in [101,202,303,404,505]:
  name=f"fewshot_{seed}";support=support_ids(rows,split["ids"]["train"],seed)
  tr=np.array([idmap[i] for i in support])
  arts,ps,logs=fit_suite(fx,y,tr,idx["val"],True,[1000+seed],states)
  val={k:part_metrics(rows,y,p,idx["val"],pairs) for k,p in ps.items()}
  regimes[name]={"train_ids":support,"balanced":True,"validation":val}
  exports[name]=arts;allpred[name]=ps;fitting[name]=logs
  print(name,"DONE",flush=True)
 methods=list(exports["fewshot_101"])
 favg={k:float(np.mean([regimes[f"fewshot_{s}"]["validation"][k]["macro_grade_mae"] for s in [101,202,303,404,505]])) for k in methods}
 fsel=min(favg,key=lambda k:(favg[k],k))
 for s in [101,202,303,404,505]:regimes[f"fewshot_{s}"]["selection"]=fsel
 # Seal choice before computing any test metrics. A later foundation prior trial can only use validation.
 choice={"selected_regime":"merged_balanced","selected_method":regimes["merged_balanced"]["selection"],"criterion":"validation macro-grade overall MAE, then overall MAE, then name","test_used_for_selection":False,"fewshot_method":fsel,"fewshot_validation_means":favg}
 write(OUT/"selection.json",choice)
 # Keep test metrics in a separate script so optional prior candidates can be added without seeing test.
 for name,ps in allpred.items():
  np.savez_compressed(OUT/(name+"_predictions.npz"),**ps)
 with gzip.open(OUT/"candidate_models.json.gz","wt",encoding="utf-8") as f:json.dump(exports,f,separators=(",",":"))
 write(OUT/"validation.json",regimes);write(OUT/"training.json",fitting)
 write(OUT/"environment.json",{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"sklearn":sklearn.__version__,"device":"cpu","threads":1,"seeds":[17,41,73],"support_seeds":[101,202,303,404,505],"epochs":400,"patience":50,"rank_coefficient":.15,"elapsed_seconds":time.perf_counter()-start,"hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"dataset.jsonl",ROOT/"split.json",ROOT/"models.py",ROOT/"data.py"]}})
 print("VALIDATION_COMPLETE",time.perf_counter()-start,flush=True)
if __name__=="__main__":main()
