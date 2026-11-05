import argparse,json,sys
from pathlib import Path
import joblib
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"scorer_v6"))
from model import predict_artifact
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);a=p.parse_args()
 raw=json.loads(a.input.read_text());single=isinstance(raw,dict);states=[raw] if single else raw
 model=joblib.load(R/"model.joblib.gz");prob=predict_artifact(model,states)
 out=[dict(zip(["overall","reasoning_depth","move_concealment"],[{"class":int(x.argmax())+1,"probabilities":x.tolist()} for x in row]),human_calibrated=False,supported_grades=[1,2,3,4]) for row in prob]
 print(json.dumps(out[0] if single else out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
