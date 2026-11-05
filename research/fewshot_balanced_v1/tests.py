import sys,json,unittest
from pathlib import Path
import numpy as np
from experiment import support_ids, prototype_fit, prototype_predict
ROOT=Path(__file__).resolve().parent
ROWS=[json.loads(x) for x in (ROOT.parent/"pilot100/dataset.jsonl").read_text().splitlines()]
SPLIT=json.loads((ROOT.parent/"scorer_v1/runs/frozen100_20261105/split.json").read_text())["ids"]
class FewShotTests(unittest.TestCase):
 def test_exact_two_unique_per_class_train_only(self):
  ids=support_ids(ROWS,SPLIT["train"],2,101)
  self.assertEqual(len(ids),8);self.assertEqual(len(set(ids)),8)
  self.assertTrue(set(ids)<=set(SPLIT["train"]))
  self.assertFalse(set(ids)&set(SPLIT["val"]+SPLIT["test"]))
  byid={r["id"]:r for r in ROWS}
  self.assertEqual([sum(byid[i]["target"]["overall"]==g for i in ids) for g in range(1,5)],[2]*4)
 def test_insufficient_class_rejected_not_duplicated(self):
  with self.assertRaises(ValueError):support_ids(ROWS,SPLIT["train"],3,101)
 def test_reproducible_support(self):
  self.assertEqual(support_ids(ROWS,SPLIT["train"],2,101),support_ids(ROWS,SPLIT["train"],2,101))
 def test_prototype_learns_support_centres(self):
  x=np.array([[0.],[0.],[10.],[10.],[20.],[20.],[30.],[30.]])
  y=np.repeat(np.arange(1,5),2)[:,None]*np.ones((1,3))
  model=prototype_fit(x,y,.001)
  pred=prototype_predict(model,np.array([[0.],[30.]]))
  np.testing.assert_allclose(pred,np.array([[1]*3,[4]*3]),atol=1e-5)


class ArtifactTests(unittest.TestCase):
 def test_frozen_data_and_selection(self):
  import hashlib
  from experiment import HASH
  self.assertEqual(hashlib.sha256((ROOT.parent/"pilot100/dataset.jsonl").read_bytes()).hexdigest(),HASH)
  report=json.loads((ROOT/"results/report.json").read_text())
  choice=json.loads((ROOT/"results/selection.json").read_text())
  self.assertEqual(report["val_ids"],SPLIT["val"]);self.assertEqual(report["test_ids"],SPLIT["test"])
  self.assertEqual(choice["method"],min(choice["validation_scores"],key=lambda k:(choice["validation_scores"][k],k)))
  self.assertFalse(choice["test_used_to_choose"])
 def test_every_export_scaler_uses_support_only(self):
  from experiment import encode_state,card_features
  models=json.loads((ROOT/"results/models.json").read_text())
  report=json.loads((ROOT/"results/report.json").read_text())
  byid={r["id"]:r for r in ROWS}
  for ep in report["episodes"]:
   support=ep["support_ids"]
   self.assertEqual(len(support),8)
   self.assertTrue(set(support)<=set(SPLIT["train"]))
   x=np.stack([encode_state(byid[i]["input"]) for i in support])
   for model in models[str(ep["seed"])].values():
    if model["kind"]=="constant":continue
    xx=card_features(x) if model.get("features")=="counts" else x
    np.testing.assert_allclose(model["mean"],xx.mean(0),atol=1e-6)
    scale=xx.std(0);scale[scale==0]=1
    np.testing.assert_allclose(model["scale"],scale,rtol=1e-5,atol=1e-6)
 def test_saved_prediction_matches_export_and_range(self):
  from experiment import prediction
  report=json.loads((ROOT/"results/report.json").read_text())
  models=json.loads((ROOT/"results/models.json").read_text())
  saved=json.loads((ROOT/"results/predictions.json").read_text())
  byid={r["id"]:r for r in ROWS}
  for seed in report["support_seeds"]:
   records=[p for p in saved if p["seed"]==seed]
   actual=prediction(models[str(seed)][report["selected_method"]],[byid[p["id"]]["input"] for p in records])
   np.testing.assert_allclose(actual,[p["prediction"] for p in records],atol=2e-5)
   self.assertTrue(np.isfinite(actual).all());self.assertTrue((actual>=1).all());self.assertTrue((actual<=4).all())

if __name__=="__main__":unittest.main()
