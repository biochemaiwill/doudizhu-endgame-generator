import unittest,json,gzip
from pathlib import Path
import numpy as np,torch
from train import R,Net,metrics
import sys
sys.path.insert(0,str(R.parent/"scorer_v2"))
from data import components
from models import encode_structured
class Checks(unittest.TestCase):
 def test_groups(self):
  rows=[json.loads(s) for s in (R/"dataset.jsonl").read_text().splitlines()]
  pairs=json.loads((R.parent/"expansion_v2/pairs.json").read_text())+json.loads((R.parent/"expansion_v3/pairs.json").read_text())
  split=json.loads((R/"split.json").read_text());owners={i:p for p,ids in split["ids"].items() for i in ids}
  self.assertEqual(len(owners),182)
  for ids in components(rows,pairs).values():self.assertEqual(len({owners[i] for i in ids}),1)
 def test_permutation_and_monotonic(self):
  row=json.loads((R/"dataset.jsonl").read_text().splitlines()[0]);state=row["input"]
  rev=dict(state,hands=[list(reversed(h)) for h in state["hands"]])
  np.testing.assert_array_equal(encode_structured(state),encode_structured(rev))
  x=torch.tensor(np.stack([encode_structured(state)]))
  for kind in ["mlp","self","cross"]:
   model=Net(kind).eval()
   with torch.no_grad():z=model(x)
   self.assertTrue(torch.all(z[:,:,1:]<=z[:,:,:-1]));self.assertEqual(z.shape,(1,3,3))
 def test_metrics_and_selection(self):
  result=json.loads((R/"results.json").read_text());v=result["validation"]
  winner=max(v,key=lambda n:(v[n]["accuracy"],v[n]["macro_recall"],-v[n]["mae"],n))
  self.assertEqual(winner,result["selected"])
  records=json.loads((R/"predictions.json").read_text());test=[r for r in records if r["split"]=="test"]
  actual=metrics(np.array([r["truth"] for r in test]),np.array([r["prediction"] for r in test]))
  self.assertEqual(actual["confusion"],result["test"][winner]["confusion"])
  self.assertAlmostEqual(actual["accuracy"],result["test"][winner]["accuracy"])
if __name__=="__main__":unittest.main()
