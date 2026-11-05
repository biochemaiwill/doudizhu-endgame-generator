import sys,os,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path[:0]=[str(D.parent/"scorer_v2_prior_trial/runtime"),str(D.parent/"scorer_v6")]
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np,torch
from tabpfn import TabPFNClassifier
from model import features
def load_model():
 meta=json.loads((D/"model.json").read_text());ctx=json.loads((D/"context.json").read_text())
 if hashlib.sha256((D/"classifier.ckpt").read_bytes()).hexdigest()!=meta["checkpoint_sha256"]:raise ValueError("checkpoint hash mismatch")
 torch.set_num_threads(1)
 m=TabPFNClassifier(model_path=D/"classifier.ckpt",device="cpu",n_estimators=2,n_jobs=1,random_state=73)
 m.fit(np.array(ctx["x"]),np.array(ctx["y"]))
 return m
def predict(states):
 m=load_model();p=m.predict_proba(np.stack([features(s,mode="base") for s in states]))
 return [{"overall":int(m.classes_[i]),"probabilities":row.tolist(),"classes":m.classes_.tolist(),"rule":"argmax"} for i,row in zip(p.argmax(1),p)]
if __name__=="__main__":
 x=json.loads(Path(sys.argv[1]).read_text());one=isinstance(x,dict)
 out=predict([x] if one else x);print(json.dumps(out[0] if one else out))
