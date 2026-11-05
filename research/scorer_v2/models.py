from pathlib import Path
from collections import Counter
import sys
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scorer_v1"))
from scorer import encode_state,card_features,OrdinalMLP
from doudizhu.moves import gen_all,gen_legal,Move,MoveType
STRUCTURED_DIM=143
def encode_structured(state):
 base=encode_state(state);extra=[]
 for hand in state["hands"]:
  moves=gen_all(Counter(hand));counts=np.zeros(15,dtype=np.float32)
  for m in moves:counts[int(m.type)]+=1
  extra.extend(np.log1p(counts).tolist())
  extra.extend([max((len(m.cards) for m in moves),default=0)/len(hand),float(any(len(m.cards)==len(hand) for m in moves)),len(set(hand))/len(hand)])
 last=state.get("last")
 lm=None if last is None else Move(MoveType(last["type"]),last["main"],last["length"],tuple(sorted(last["cards"])))
 hand=state["hands"][state["turn"]];legal=gen_legal(Counter(hand),lm)
 extra.extend([np.log1p(len(legal)),max((len(m.cards) for m in legal),default=0)/len(hand),float(any(len(m.cards)==len(hand) for m in legal))])
 return np.concatenate([base,np.asarray(extra,dtype=np.float32)]).astype(np.float32)
class RoleOrdinalNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.hand_encoder=nn.Sequential(nn.Linear(15,32),nn.GELU(),nn.Linear(32,16),nn.GELU())
  self.body=nn.Sequential(nn.Linear(48+STRUCTURED_DIM-45,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),nn.GELU(),nn.Linear(32,3))
  self.cut_base=nn.Parameter(torch.full((3,1),-1.));self.cut_steps=nn.Parameter(torch.zeros(3,2))
 def forward(self,x):
  hands=self.hand_encoder(x[:,:45].reshape(-1,3,15)).flatten(1)
  scores=self.body(torch.cat([hands,x[:,45:]],1))
  cuts=torch.cat([self.cut_base,self.cut_base+torch.cumsum(nn.functional.softplus(self.cut_steps),1)],1)
  return scores[:,:,None]-cuts[None,:,:]
 def scores(self,x):return 1+torch.sigmoid(self(x)).sum(-1)
def export_torch(model,kind,scaler):
 dim=86 if kind=="ordinal_mlp" else STRUCTURED_DIM
 return {"kind":kind,"features":"board" if dim==86 else "structured","feature_dim":dim,"mean":np.asarray(scaler.mean_).tolist(),"scale":np.asarray(scaler.scale_).tolist(),"weights":{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}}
def _features(kind,states):
 if kind=="structured":return np.stack([encode_structured(s) for s in states])
 x=np.stack([encode_state(s) for s in states])
 return card_features(x) if kind=="counts" else x
def predict_artifact(a,states):
 if not states:return np.empty((0,3),dtype=np.float32)
 kind=a["kind"]
 if kind=="constant":return np.clip(np.tile(np.asarray(a["mean"]),(len(states),1)),1,4)
 x=_features(a.get("features","board"),states)
 if kind=="ridge":
  z=(x-np.asarray(a["mean"]))/np.asarray(a["scale"]);y=z@np.asarray(a["coef"]).T+np.asarray(a["intercept"])
 elif kind=="tree_ensemble":
  total=np.zeros((len(states),3))
  for tree in a["trees"]:
   nodes=np.zeros(len(states),dtype=np.int64)
   left=np.asarray(tree["children_left"]);right=np.asarray(tree["children_right"]);feature=np.asarray(tree["feature"]);threshold=np.asarray(tree["threshold"])
   while True:
    active=np.flatnonzero(left[nodes]!=-1)
    if len(active)==0:break
    n=nodes[active];go=x[active,feature[n]]<=threshold[n];nodes[active]=np.where(go,left[n],right[n])
   total+=np.asarray(tree["value"])[nodes].reshape(len(states),3)
  y=total/len(a["trees"])
 elif kind in ("ordinal_mlp","structured_mlp","role_ordinal"):
  m=RoleOrdinalNet() if kind=="role_ordinal" else OrdinalMLP(input_dim=a.get("feature_dim",86 if kind=="ordinal_mlp" else STRUCTURED_DIM))
  m.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in a["weights"].items()});m.eval()
  z=(x-np.asarray(a["mean"]))/np.asarray(a["scale"])
  with torch.no_grad():y=m.scores(torch.tensor(z,dtype=torch.float32)).numpy()
 else:raise ValueError("unknown artifact kind: "+kind)
 return np.clip(y,1,4)
