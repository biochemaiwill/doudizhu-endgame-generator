import os
os.environ["OMP_NUM_THREADS"]="1"
import sys,json,gzip,copy,hashlib,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"scorer_v2"))
from data import components,subset,class_weights
from models import encode_structured,OrdinalMLP
torch.set_num_threads(1)
KEYS=["overall","reasoning_depth","move_concealment"]
class Net(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  if kind=="mlp":self.m=OrdinalMLP(143)
  else:
   self.rank=nn.Embedding(15,64);self.count=nn.Embedding(5,64);self.seat=nn.Embedding(3,64);self.turn=nn.Embedding(3,64)
   layer=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True)
   self.enc=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   self.cross=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
   self.norm=nn.LayerNorm(64);self.head=nn.Sequential(nn.Linear(64+98,64),nn.GELU(),nn.Linear(64,3))
   self.base=nn.Parameter(torch.full((3,1),-1.));self.steps=nn.Parameter(torch.zeros(3,2))
 def forward(self,x):
  if self.kind=="mlp":return self.m(x)
  counts=(x[:,:45].reshape(-1,3,15)*4).round().long().clamp(0,4)
  turn=x[:,45:48].argmax(1)
  z=self.count(counts)+self.rank(torch.arange(15,device=x.device))[None,None]+self.seat(torch.arange(3,device=x.device))[None,:,None]+self.turn(turn)[:,None,None]
  if self.kind=="self":z=self.enc(z.flatten(1,2))
  else:
   z=self.enc(z.reshape(-1,15,64)).reshape(-1,3,15,64)
   out=[]
   for s in range(3):
    others=torch.cat([z[:,j] for j in range(3) if j!=s],1)
    out.append(self.norm(z[:,s]+self.cross(z[:,s],others,others,need_weights=False)[0]))
   z=torch.cat(out,1)
  h=self.head(torch.cat([z.mean(1),x[:,45:]],1))
  cuts=torch.cat([self.base,self.base+torch.cumsum(nn.functional.softplus(self.steps),1)],1)
  return h[:,:,None]-cuts[None]
 def scores(self,x):return 1+torch.sigmoid(self(x)).sum(-1)
def metrics(y,p):
 pred=np.floor(p+.5).clip(1,4).astype(int);actual=y.astype(int)
 cm=[[int(np.sum((actual[:,0]==a)&(pred[:,0]==b))) for b in range(1,5)] for a in range(1,5)]
 rec=[cm[i][i]/sum(cm[i]) if sum(cm[i]) else None for i in range(4)]
 return dict(accuracy=float(np.mean(pred[:,0]==actual[:,0])),macro_recall=float(np.mean([v for v in rec if v is not None])),recall=rec,confusion=cm,mae=float(np.abs(y[:,0]-p[:,0]).mean()),head_accuracy=[float(np.mean(pred[:,h]==actual[:,h])) for h in range(3)])
def run():
 if (R/"results.json").exists():raise SystemExit("refuse overwrite")
 start=time.time()
 rows=[]
 for p in [R.parent/"scorer_v2/dataset.jsonl",R.parent/"expansion_v3/dataset.jsonl"]:rows += [json.loads(s) for s in p.read_text().splitlines()]
 pairs=json.loads((R.parent/"expansion_v2/pairs.json").read_text())+json.loads((R.parent/"expansion_v3/pairs.json").read_text())
 groups=components(rows,pairs);byid={r["id"]:r for r in rows}
 gc={g:tuple(sum(byid[i]["target"]["overall"]==k for i in ids) for k in range(1,5)) for g,ids in groups.items()}
 eligible={g:ids for g,ids in groups.items() if not any(byid[i]["provenance"]["source"]=="MasterMind train API" for i in ids)}
 rng=np.random.default_rng(2026110516)
 for _ in range(500):
  va=subset(eligible,gc,[9,9,9,2],rng)
  if va is None:continue
  te=subset({g:ids for g,ids in eligible.items() if g not in va},gc,[9,9,9,2],rng)
  if te is not None:break
 else:raise ValueError("split infeasible")
 assignment={g:"val" if g in va else "test" if g in te else "train" for g in groups}
 parts={p:sorted(i for g,ids in groups.items() if assignment[g]==p for i in ids) for p in ["train","val","test"]}
 index={r["id"]:i for i,r in enumerate(rows)};idx={p:np.array([index[i] for i in ids]) for p,ids in parts.items()}
 for ids in groups.values():assert len({p for p in parts if set(ids)&set(parts[p])})==1
 split=dict(seed=2026110516,ids=parts,groups=groups,assignment=assignment,counts={p:{str(g):sum(byid[i]["target"]["overall"]==g for i in ids) for g in range(1,5)} for p,ids in parts.items()})
 (R/"split.json").write_text(json.dumps(split,indent=2))
 (R/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
 x=np.stack([encode_structured(r["input"]) for r in rows]);y=np.array([[r["target"][k] for k in KEYS] for r in rows],np.float32)
 tx=torch.tensor(x);target=torch.tensor((y[:,:,None]>np.arange(1,4)[None,None]).astype(np.float32))
 registry={};predictions={};validation={};history={}
 for kind in ["mlp","self","cross"]:
  for balanced in [False,True]:
   name=kind+("_balanced" if balanced else "_natural");seedpred=[];weights=[];logs=[]
   tr,vi=idx["train"],idx["val"];w=torch.tensor(class_weights(y[tr,0]) if balanced else np.ones(len(tr),np.float32))
   for seed in [17,41,73]:
    torch.manual_seed(seed);model=Net(kind);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    best=None;bestkey=None;bestepoch=0
    for epoch in range(1,251):
     model.train();opt.zero_grad();logits=model(tx[tr]);loss=(nn.functional.binary_cross_entropy_with_logits(logits,target[tr],reduction="none").mean((1,2))*w).sum()/w.sum()
     loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.eval()
     with torch.no_grad():pv=model.scores(tx[vi]).numpy()
     m=metrics(y[vi],pv);key=(m["accuracy"],m["macro_recall"],-m["mae"])
     if bestkey is None or key>bestkey:bestkey=key;best=copy.deepcopy(model.state_dict());bestepoch=epoch
     if epoch-bestepoch>=35:break
    model.load_state_dict(best);model.eval()
    with torch.no_grad():seedpred.append(model.scores(tx).numpy())
    weights.append({k:v.tolist() for k,v in best.items()});logs.append(dict(seed=seed,best_epoch=bestepoch,stopped_epoch=epoch))
   predictions[name]=np.mean(seedpred,0);registry[name]=dict(kind=kind,weights=weights)
   validation[name]=metrics(y[idx["val"]],predictions[name][idx["val"]]);history[name]=logs
   print(name,validation[name],flush=True)
 selected=max(validation,key=lambda n:(validation[n]["accuracy"],validation[n]["macro_recall"],-validation[n]["mae"],n))
 (R/"selection.json").write_text(json.dumps(dict(selected=selected,criterion="val accuracy, macro recall, negative MAE",test_used=False),indent=2))
 test={n:metrics(y[idx["test"]],p[idx["test"]]) for n,p in predictions.items()}
 with gzip.open(R/"model.json.gz","wt") as f:json.dump(registry[selected],f)
 # Reload all selected parameters and compare predictions.
 art=registry[selected];ps=[]
 for state in art["weights"]:
  m=Net(art["kind"]);m.load_state_dict({k:torch.tensor(v) for k,v in state.items()});m.eval()
  with torch.no_grad():ps.append(m.scores(tx).numpy())
 assert np.allclose(np.mean(ps,0),predictions[selected],atol=1e-6)
 result=dict(selected=selected,validation=validation,test=test,history=history,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device="cpu",threads=1),seconds=time.time()-start,classification_rule="floor(expected ordinal score + .5), clipped 1..4",split=split["counts"],limit="previously exposed weak-label development corpus, not independent human test")
 (R/"results.json").write_text(json.dumps(result,indent=2))
 (R/"predictions.json").write_text(json.dumps([dict(id=rows[i]["id"],truth=y[i].tolist(),prediction=predictions[selected][i].tolist(),split=p) for p in parts for i in idx[p]],indent=2))
 print("FINAL",selected,test[selected],split["counts"],flush=True)
if __name__=="__main__":run()
