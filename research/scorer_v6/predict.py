import argparse,json,joblib,torch
from pathlib import Path
from model import predict_artifact
R=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--model",type=Path,default=R/"model.joblib.gz");a=p.parse_args()
 torch.set_num_threads(1);raw=json.loads(a.input.read_text());one=isinstance(raw,dict);states=[raw] if one else raw
 probs=predict_artifact(joblib.load(a.model),states)
 out=[{"overall":int(p[0].argmax())+1,"reasoning_depth":int(p[1].argmax())+1,"move_concealment":int(p[2].argmax())+1,"probabilities":{name:v.tolist() for name,v in zip(["overall","reasoning_depth","move_concealment"],p)},"supported_grades":[1,2,3,4],"human_calibrated":False} for p in probs]
 print(json.dumps(out[0] if one else out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
