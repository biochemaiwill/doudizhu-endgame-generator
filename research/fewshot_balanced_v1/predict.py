"""Score boards using the balanced few-shot artifact; no search at inference."""
import argparse,json
from pathlib import Path
import torch
from experiment import prediction
def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--model",type=Path,default=Path(__file__).parent/"results/model.json")
 a=p.parse_args();torch.set_num_threads(1)
 payload=json.loads(a.input.read_text());one=isinstance(payload,dict);states=[payload] if one else payload
 model=json.loads(a.model.read_text());values=prediction(model["artifact"],states)
 out=[dict(zip(("overall","reasoning_depth","move_concealment"),map(float,row)),supported_grades=[1,2,3,4],grade_5="unsupported_no_training_examples",human_calibrated=False,ready_for_generation_reward=False,model=model["selected_name"]) for row in values]
 print(json.dumps(out[0] if one else out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
