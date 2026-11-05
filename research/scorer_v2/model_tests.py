import unittest,copy,json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models import *
from models import _features
class ModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  rows=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/"expansion_v2/dataset.jsonl").read_text().splitlines()]
  cls.states=[r["input"] for r in rows[:4]]
 def test_permuted_card_order(self):
  s=copy.deepcopy(self.states[0]);s["hands"]=[list(reversed(h)) for h in s["hands"]]
  np.testing.assert_array_equal(encode_structured(s),encode_structured(self.states[0]))
 def test_feature_shape_and_no_solver(self):
  before=set(sys.modules);x=encode_structured(self.states[0])
  self.assertEqual(x.shape,(STRUCTURED_DIM,));self.assertTrue(np.isfinite(x).all())
  self.assertFalse(any(k.startswith("endgame") for k in set(sys.modules)-before))
 def test_role_logits_are_monotone(self):
  m=RoleOrdinalNet();m.eval();x=torch.randn(8,STRUCTURED_DIM)
  with torch.no_grad():z=m(x);s=m.scores(x)
  self.assertEqual(tuple(z.shape),(8,3,3));self.assertTrue((z[:,:,1:]<=z[:,:,:-1]).all());self.assertTrue(((s>=1)&(s<=4)).all())
 def test_torch_export_reloads_exactly(self):
  for kind in ("ordinal_mlp","structured_mlp","role_ordinal"):
   x=_features("board" if kind=="ordinal_mlp" else "structured",self.states);sc=StandardScaler().fit(x)
   m=RoleOrdinalNet() if kind=="role_ordinal" else OrdinalMLP(x.shape[1]);m.eval()
   with torch.no_grad():expected=m.scores(torch.tensor(sc.transform(x),dtype=torch.float32)).numpy()
   np.testing.assert_allclose(predict_artifact(export_torch(m,kind,sc),self.states),expected,atol=2e-6)
 def test_tree_traversal(self):
  tree={"children_left":[1,-1,-1],"children_right":[2,-1,-1],"feature":[0,-2,-2],"threshold":[1.,-2.,-2.],"value":[[2.,2.,2.],[1.,1.,1.],[4.,4.,4.]]}
  out=predict_artifact({"kind":"tree_ensemble","features":"counts","trees":[tree]},self.states)
  np.testing.assert_array_equal(out,np.full((4,3),4.))
if __name__=="__main__":unittest.main()
