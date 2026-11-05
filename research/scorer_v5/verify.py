import sys,json,gzip
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from run import Net,metrics,encode_structured
torch.set_num_threads(1)
rows=[json.loads(s) for s in (R.parent/'scorer_v4/dataset.jsonl').read_text().splitlines()]
split=json.loads((R.parent/'scorer_v4/split.json').read_text())['ids'];lookup={r['id']:i for i,r in enumerate(rows)}
x=torch.tensor(np.stack([encode_structured(r['input']) for r in rows]));y=np.array([[r['target'][k] for k in ['overall','reasoning_depth','move_concealment']] for r in rows])
models=json.load(gzip.open(R/'models.json.gz','rt'));saved=json.loads((R/'predictions.json').read_text());result=json.loads((R/'results.json').read_text());out={}
for arm,weights in models.items():
 ps=[];seedmetrics=[]
 for seed,w in zip([17,41,73],weights):
  m=Net('cross');m.load_state_dict({k:torch.tensor(v) for k,v in w.items()});m.eval()
  with torch.no_grad():p=m.scores(x).numpy()
  ps.append(p);seedmetrics.append({'seed':seed,**{part:metrics(y[[lookup[i] for i in split[part]]],p[[lookup[i] for i in split[part]]]) for part in ['val','test']}})
 mean=np.mean(ps,0);assert np.allclose(mean,saved[arm],atol=1e-6)
 for part in ['val','test']:
  ix=[lookup[i] for i in split[part]];actual=metrics(y[ix],mean[ix]);expected=result['validation' if part=='val' else 'test'][arm]
  assert actual['confusion']==expected['confusion']
  assert abs(actual['accuracy']-expected['accuracy'])<1e-8
 out[arm]=seedmetrics
(R/'verification.json').write_text(json.dumps({'export_parity':True,'metric_parity':True,'per_seed':out},indent=2))
print('PASS 6 model reloads, ensemble parity, metric recomputation')
