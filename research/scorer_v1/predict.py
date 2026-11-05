"""Score one board JSON or a list of boards; no solver is called."""
import json,argparse
from pathlib import Path
import torch
from scorer import predict_export
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--model",type=Path,default=Path(__file__).parent/"runs/frozen100_20261105/model.json")
 a=p.parse_args();torch.set_num_threads(1)
 payload=json.loads(a.input.read_text());one=isinstance(payload,dict);states=[payload] if one else payload
 model=json.loads(a.model.read_text());pred=predict_export(model["artifact"],states)
 out=[{"overall":float(v[0]),"reasoning_depth":float(v[1]),"move_concealment":float(v[2]),"supported_grades":[1,2,3,4],"grade_5":"unsupported_no_training_examples","human_calibrated":False,"ready_for_generation_reward":False,"model":model["selected_name"]} for v in pred]
 print(json.dumps(out[0] if one else out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
