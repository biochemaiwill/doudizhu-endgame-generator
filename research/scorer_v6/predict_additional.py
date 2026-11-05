import json,argparse,joblib
from pathlib import Path
import numpy as np
from model import features
R=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);a=p.parse_args()
 raw=json.loads(a.input.read_text());one=isinstance(raw,dict);states=[raw] if one else raw
 model=joblib.load(R/"additional_model.joblib.gz");x=np.stack([features(s,model["feature_mode"]) for s in states])
 if model["scaler"] is not None:x=model["scaler"].transform(x)
 ys=model["model"].predict(x)+1
 out=[{"overall":int(y),"auxiliary_heads":"not_trained","model":model["config"]} for y in ys]
 print(json.dumps(out[0] if one else out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
