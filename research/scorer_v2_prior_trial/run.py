import sys,os,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"runtime"));sys.path.insert(0,str(ROOT.parent/"scorer_v2"))
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np,torch
from tabpfn import TabPFNRegressor
from models import encode_structured
torch.set_num_threads(1)
rows=[json.loads(s) for s in (ROOT.parent/"scorer_v2/dataset.jsonl").read_text().splitlines()]
byid={r["id"]:r for r in rows};split=json.loads((ROOT.parent/"scorer_v2/split.json").read_text())["ids"]
tr=split["train"];va=split["val"]
X=np.stack([encode_structured(byid[i]["input"]) for i in tr])
V=np.stack([encode_structured(byid[i]["input"]) for i in va])
Y=np.array([[byid[i]["target"][k] for k in ["overall","reasoning_depth","move_concealment"]] for i in tr])
truth=np.array([[byid[i]["target"][k] for k in ["overall","reasoning_depth","move_concealment"]] for i in va])
out={"train_ids":tr,"val_ids":va,"models":{},"test_evaluated":False,"checkpoint_sha256":hashlib.sha256((ROOT/"model.ckpt").read_bytes()).hexdigest(),"package":"tabpfn==2.0.9","sklearn":"1.6.1"}
start=time.monotonic()
for features in ["board","structured"]:
 pred=[]
 for h in range(3):
  reg=TabPFNRegressor(model_path=ROOT/"model.ckpt",device="cpu",n_estimators=2,n_jobs=1,random_state=73)
  reg.fit(X[:,:86] if features=="board" else X,Y[:,h])
  pred.append(reg.predict(V[:,:86] if features=="board" else V))
 p=np.clip(np.array(pred).T,1,4)
 macro=float(np.mean([np.abs(p[truth[:,0]==g,0]-g).mean() for g in range(1,5)]))
 out["models"][features]={"prediction":p.tolist(),"overall_mae":float(np.abs(p[:,0]-truth[:,0]).mean()),"macro_grade_mae":macro,"config":{"n_estimators":2,"random_state":73,"feature_mode":features}}
 (ROOT/"validation.json").write_text(json.dumps(out,indent=2))
 print(features,macro,flush=True)
out["seconds"]=time.monotonic()-start
(ROOT/"validation.json").write_text(json.dumps(out,indent=2))
