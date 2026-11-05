import argparse,json,gzip
from pathlib import Path
import numpy as np,torch
from train import Net,R
from models import encode_structured
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);a=p.parse_args()
 x=json.loads(a.input.read_text());states=[x] if isinstance(x,dict) else x
 with gzip.open(R/"model.json.gz","rt") as f:art=json.load(f)
 features=torch.tensor(np.stack([encode_structured(s) for s in states]));pred=[]
 for w in art["weights"]:
  m=Net(art["kind"]);m.load_state_dict({k:torch.tensor(v) for k,v in w.items()});m.eval()
  with torch.no_grad():pred.append(m.scores(features).numpy())
 values=np.mean(pred,0);labels=np.floor(values+.5).clip(1,4).astype(int)
 print(json.dumps([dict(scores=dict(zip(["overall","reasoning_depth","move_concealment"],map(float,v))),classes=dict(zip(["overall","reasoning_depth","move_concealment"],map(int,c))),unsupported=[5]) for v,c in zip(values,labels)],indent=2))
if __name__=="__main__":main()
