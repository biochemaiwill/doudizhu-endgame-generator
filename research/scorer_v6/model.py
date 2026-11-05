"""Fixed CE/argmax models and state-only features."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"scorer_v2"))
from models import encode_structured

def features(state,mode="base"):
 x=encode_structured(state)
 if mode=="base":return x
 h=[np.bincount(a,minlength=18)[3:18] for a in state["hands"]];extra=[]
 for a in range(3):
  for b in range(3):
   if a==b:continue
   for n in [1,2,3,4]:
    aa=np.flatnonzero(h[a]>=n);bb=np.flatnonzero(h[b]>=n)
    ma=aa.max() if len(aa) else -1;mb=bb.max() if len(bb) else -1
    extra.extend([(ma-mb)/15.,sum(aa>mb)/max(1,len(aa))])
 return np.r_[x,np.asarray(extra,dtype=np.float32)].astype(np.float32)
class CENet(nn.Module):
 def __init__(self,kind,dim):
  super().__init__();self.kind=kind
  if kind=="mlp":self.body=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),nn.GELU(),nn.Linear(32,12))
  else:
   self.rank=nn.Embedding(15,32);self.count=nn.Embedding(5,32);self.seat=nn.Embedding(3,32)
   layer=nn.TransformerEncoderLayer(32,4,64,.1,batch_first=True,norm_first=True)
   self.enc=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
   self.cross=nn.MultiheadAttention(32,4,dropout=.1,batch_first=True)
   self.norm=nn.LayerNorm(32)
   self.body=nn.Sequential(nn.Linear(32+dim-45,64),nn.GELU(),nn.Linear(64,12))
 def forward(self,x):
  if self.kind=="mlp":return self.body(x).reshape(-1,3,4)
  c=(x[:,:45].reshape(-1,3,15)*4).round().long().clamp(0,4)
  z=self.count(c)+self.rank(torch.arange(15,device=x.device))[None,None]+self.seat(torch.arange(3,device=x.device))[None,:,None]
  z=self.enc(z.reshape(-1,15,32)).reshape(-1,3,15,32);out=[]
  for s in range(3):
   other=torch.cat([z[:,t] for t in range(3) if t!=s],1)
   out.append(self.norm(z[:,s]+self.cross(z[:,s],other,other,need_weights=False)[0]))
  z=torch.cat(out,1).mean(1)
  return self.body(torch.cat([z,x[:,45:]],1)).reshape(-1,3,4)
def metric(y,p):
 pred=p.argmax(-1);cm=np.zeros((4,4),int)
 for a,b in zip(y[:,0],pred[:,0]):cm[a,b]+=1
 recall=[float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i in range(4)]
 return dict(accuracy=float((pred[:,0]==y[:,0]).mean()),macro_recall=float(np.mean([v for v in recall if v is not None])),recall=recall,confusion=cm.tolist(),head_accuracy=[float((pred[:,i]==y[:,i]).mean()) for i in range(3)])
def predict_artifact(a,states):
 x=np.stack([features(s,a["mode"]) for s in states])
 if a["kind"]=="neural":
  z=x.copy()
  # Preserve literal count tokens for cross model, standardize its summary only.
  start=45 if a["net"]=="cross" else 0
  z[:,start:]=(z[:,start:]-np.array(a["mean"])) / np.array(a["scale"])
  m=CENet(a["net"],x.shape[1]);m.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in a["weights"].items()});m.eval()
  with torch.no_grad():return torch.softmax(m(torch.tensor(z,dtype=torch.float32)),-1).numpy()
 result=np.zeros((len(states),3,4))
 for h,est in enumerate(a["estimators"]):
  pp=est.predict_proba(x)
  for j,c in enumerate(est.classes_):result[:,h,int(c)]=pp[:,j]
 return result
