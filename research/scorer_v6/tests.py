import unittest,json
from pathlib import Path
import numpy as np,torch,joblib
from model import CENet,features,predict_artifact,metric
import importlib.util
_spec=importlib.util.spec_from_file_location('ce_training',Path(__file__).resolve().parent/'train.py')
_training=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_training)
make_data=_training.make_data
R=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_groups_and_all_classes(self):
  rows,y,folds,groups=make_data();owner={i:g for g,ids in groups.items() for i in ids}
  self.assertEqual(len(rows),189)
  for fold in folds:
   self.assertFalse({owner[rows[i]["id"]] for i in fold["tr"]}&{owner[rows[i]["id"]] for i in fold["va"]})
   self.assertEqual(set(y[fold["va"],0]),{0,1,2,3})
 def test_permutation_and_ce(self):
  state={"hands":[[3,4,4],[5,5,6],[7,8]],"turn":0,"last":None,"passes":0}
  other={**state,"hands":[list(reversed(h)) for h in state["hands"]]}
  np.testing.assert_array_equal(features(state,"relational"),features(other,"relational"))
  x=torch.tensor(np.stack([features(state,"relational")]))
  for kind in ["mlp","cross"]:
   m=CENet(kind,len(x[0]));out=m(x)
   self.assertEqual(tuple(out.shape),(1,3,4))
   loss=torch.nn.functional.cross_entropy(out[:,0],torch.tensor([3]));loss.backward()
   self.assertTrue(torch.isfinite(loss))
 def test_probabilities_and_argmax(self):
  rows=json.loads((R/"dataset.json").read_text());model=joblib.load(R/"model.joblib.gz")
  p=predict_artifact(model,[r["input"] for r in rows[:8]])
  np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
  self.assertTrue((p>=0).all())
  choice=json.loads((R/"selection.json").read_text())
  scores=json.loads((R/"cv_results.json").read_text())
  best=max(scores,key=lambda n:(scores[n]["mean_accuracy"],scores[n]["mean_macro_recall"],n))
  self.assertEqual(best,choice["name"]);self.assertFalse(choice["external_test_read"])
if __name__=="__main__":unittest.main()
