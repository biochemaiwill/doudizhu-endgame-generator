"""Small CPU-only board encoders and ordinal model. No solving during prediction."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pilot-v1/vendor'))
from doudizhu.moves import gen_all
import numpy as np
import torch
from torch import nn

FEATURE_DIM=86
SUPPORTED_GRADES=(1,2,3,4)
def encode_state(state):
 hands=state["hands"];turn=state["turn"];passes=state["passes"];last=state.get("last")
 if len(hands)!=3 or turn not in range(3) or passes not in (0,1):raise ValueError("invalid seats/pass count")
 if any(not h for h in hands):raise ValueError("terminal state cannot be rated")
 if last is None and passes:raise ValueError("free lead cannot have passes")
 all_cards=[c for h in hands for c in h]
 if last is not None:
  if not last.get("cards") or last["type"] not in range(1,15):raise ValueError("invalid last move")
  if last["main"] not in range(3,18) or not 1<=last["length"]<=12:raise ValueError("invalid last rank/length")
  shapes=gen_all(Counter(last["cards"]))
  if not any(tuple(sorted(last["cards"]))==m.cards and int(m.type)==last["type"] and m.main==last["main"] and m.length==last["length"] for m in shapes):raise ValueError("inconsistent last card type")
  all_cards+=last["cards"]
 if any(type(c) is not int or not 3<=c<=17 for c in all_cards):raise ValueError("invalid rank")
 if any(n>(1 if c>=16 else 4) for c,n in Counter(all_cards).items()):raise ValueError("more cards than one deck")
 counts=np.zeros((3,15),dtype=np.float32)
 for i,h in enumerate(hands):
  for c in h:counts[i,c-3]+=0.25
 who=np.eye(3,dtype=np.float32)[turn]
 lasttype=np.zeros(15,dtype=np.float32);lastcounts=np.zeros(15,dtype=np.float32)
 owner=np.zeros(3,dtype=np.float32)
 if last is None:lasttype[0]=1;rank=length=0.
 else:
  lasttype[last["type"]]=1
  for c in last["cards"]:lastcounts[c-3]+=0.25
  rank=last["main"]/17.;length=last["length"]/12.
  owner[(turn-1-passes)%3]=1
 return np.concatenate([counts.ravel(),who,lasttype,lastcounts,
     np.array([rank,length],dtype=np.float32),np.eye(2,dtype=np.float32)[passes],
     np.array([float(last is None)],dtype=np.float32),owner])

def split_rows(rows,seed):
 if len(rows)!=100:raise ValueError("frozen pilot split expects 100 rows")
 targets={"train":[30,27,11,2],"val":[6,6,2,1],"test":[6,6,2,1]}
 fixed=[r for r in rows if r["provenance"]["source"]=="MasterMind train API"
        or r["provenance"]["family"]=="gamersky-201411-487845"]
 parts={"train":[r["id"] for r in fixed],"val":[],"test":[]}
 fixedids=set(parts["train"]);remaining=[r for r in rows if r["id"] not in fixedids]
 families=Counter(r["provenance"]["family"] for r in remaining)
 if any(n>1 for n in families.values()):raise ValueError("additional multi-row families require group split")
 rng=np.random.default_rng(seed)
 for grade in range(1,5):
  bucket=sorted([r["id"] for r in remaining if r["target"]["overall"]==grade])
  rng.shuffle(bucket)
  nfixed=sum(r["target"]["overall"]==grade for r in fixed)
  needs=[targets["train"][grade-1]-nfixed,targets["val"][grade-1],targets["test"][grade-1]]
  if min(needs)<0 or sum(needs)!=len(bucket):raise ValueError("stratification incompatible with grouping")
  offset=0
  for part,n in zip(("train","val","test"),needs):
   parts[part].extend(bucket[offset:offset+n]);offset+=n
 for part in parts:parts[part]=sorted(parts[part])
 return parts

class OrdinalMLP(nn.Module):
 def __init__(self,input_dim=FEATURE_DIM):
  super().__init__()
  self.body=nn.Sequential(nn.Linear(input_dim,32),nn.ReLU(),nn.Dropout(.2),
                          nn.Linear(32,16),nn.ReLU(),nn.Linear(16,3))
  self.cut_base=nn.Parameter(torch.full((3,1),-1.))
  self.cut_steps=nn.Parameter(torch.zeros(3,2))
 def forward(self,x):
  cuts=torch.cat([self.cut_base,self.cut_base+torch.cumsum(nn.functional.softplus(self.cut_steps),dim=1)],dim=1)
  return self.body(x).unsqueeze(-1)-cuts.unsqueeze(0)
 def scores(self,x):
  return 1+torch.sigmoid(self(x)).sum(-1)

def card_features(x):
 return x[:,:45].reshape(-1,3,15).sum(axis=2)*4

def predict_export(artifact,states):
 x=np.stack([encode_state(s) for s in states])
 kind=artifact["kind"]
 if kind=="constant":y=np.tile(np.asarray(artifact["mean"]),(len(states),1))
 elif kind=="ridge":
  z=card_features(x) if artifact["features"]=="counts" else x
  z=(z-np.asarray(artifact["mean"]))/np.asarray(artifact["scale"])
  y=z@np.asarray(artifact["coef"]).T+np.asarray(artifact["intercept"])
 elif kind=="ordinal_ensemble":
  z=(x-np.asarray(artifact["mean"]))/np.asarray(artifact["scale"])
  preds=[]
  for weights in artifact["weights"]:
   model=OrdinalMLP()
   model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in weights.items()})
   model.eval()
   with torch.no_grad():preds.append(model.scores(torch.tensor(z,dtype=torch.float32)).numpy())
  y=np.mean(preds,axis=0)
 else:raise ValueError("unknown model kind")
 return np.clip(y,1,4)
