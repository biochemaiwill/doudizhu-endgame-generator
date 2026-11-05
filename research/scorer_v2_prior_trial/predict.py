"""CPU inference for the selected TabPFN feature configuration; no labels from input."""
import os,sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"runtime"));sys.path.insert(0,str(ROOT.parent/"scorer_v2"))
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np,torch
from tabpfn import TabPFNRegressor
from models import encode_structured
def predict(states,features="structured"):
 meta=json.loads((ROOT/"validation.json").read_text())
 if hashlib.sha256((ROOT/"model.ckpt").read_bytes()).hexdigest()!=meta["checkpoint_sha256"]:raise ValueError("checkpoint changed")
 rows={r["id"]:r for r in map(json.loads,(ROOT.parent/"scorer_v2/dataset.jsonl").read_text().splitlines())}
 x=np.stack([encode_structured(rows[i]["input"]) for i in meta["train_ids"]])
 y=np.array([[rows[i]["target"][k] for k in ("overall","reasoning_depth","move_concealment")] for i in meta["train_ids"]])
 z=np.stack([encode_structured(s) for s in states])
 if features=="board":x=x[:,:86];z=z[:,:86]
 torch.set_num_threads(1);p=[]
 for h in range(3):
  reg=TabPFNRegressor(model_path=ROOT/"model.ckpt",device="cpu",n_estimators=2,n_jobs=1,random_state=73)
  reg.fit(x,y[:,h]);p.append(reg.predict(z))
 return np.clip(np.array(p).T,1,4)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--features",default="structured",choices=["structured","board"]);a=p.parse_args()
 x=json.loads(a.input.read_text());one=isinstance(x,dict)
 print(json.dumps(predict([x] if one else x,a.features).tolist()))
