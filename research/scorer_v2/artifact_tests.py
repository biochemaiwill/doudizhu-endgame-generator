import json,gzip,unittest,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
class ArtifactTests(unittest.TestCase):
 def test_group_split_artifact_matches_builder(self):
  from data import load_rows,make_split
  rows=load_rows();pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
  self.assertEqual(make_split(rows,pairs),json.loads((ROOT/"split.json").read_text()))
 def test_every_scaler_support_only(self):
  from models import encode_state,encode_structured,card_features
  rows=list(map(json.loads,(ROOT/"dataset.jsonl").read_text().splitlines()));lookup={r["id"]:i for i,r in enumerate(rows)}
  board=np.stack([encode_state(r["input"]) for r in rows]);structured=np.stack([encode_structured(r["input"]) for r in rows])
  features={"board":board,"structured":structured,"counts":card_features(board)}
  v=json.loads((ROOT/"run/validation.json").read_text())
  with gzip.open(ROOT/"run/candidate_models.json.gz","rt") as f:models=json.load(f)
  def visit(a,ids):
   if a["kind"]=="finish_gate":visit(a["base"],ids)
   elif a["kind"]=="ensemble":
    for child in a["members"]:visit(child,ids)
   elif a["kind"] not in ["tree_ensemble","constant"]:
    ix=[lookup[i] for i in ids]
    np.testing.assert_allclose(a["mean"],features[a["features"]][ix].mean(0),atol=1e-6)
  for name,arts in models.items():
   for a in arts.values():visit(a,v[name]["train_ids"])
 def test_final_selection_is_validation_argmin(self):
  choice=json.loads((ROOT/"run/selection_final.json").read_text())
  val=choice["validation"];winner=min(val,key=lambda k:(val[k]["macro_grade_mae"],val[k]["mae"]["overall"],k))
  self.assertEqual(choice["selected"],winner);self.assertFalse(choice["test_used_for_selection"])
 def test_prior_context_train_only(self):
  p=json.loads((ROOT.parent/"scorer_v2_prior_trial/validation.json").read_text())
  split=json.loads((ROOT/"split.json").read_text())
  self.assertEqual(p["train_ids"],split["ids"]["train"]);self.assertEqual(p["val_ids"],split["ids"]["val"])
  self.assertFalse(set(p["train_ids"])&set(split["ids"]["test"]))
 def test_report_metrics_recompute(self):
  report=json.loads((ROOT/"run/test_report.json").read_text());pred=json.loads((ROOT/"run/predictions_test_final.json").read_text())
  y=np.array([x["truth"][0] for x in pred]);p=np.array([x["prediction"][0] for x in pred])
  result=report["comparisons"][report["selected"]]
  self.assertAlmostEqual(float(np.abs(y-p).mean()),result["mae"]["overall"],places=6)
  macro=np.mean([np.abs(y[y==g]-p[y==g]).mean() for g in [1,2,3,4]])
  self.assertAlmostEqual(float(macro),result["macro_grade_mae"],places=6)
  natural=[x for x in pred if not x["engineered_control"]]
  self.assertAlmostEqual(np.mean([abs(x["truth"][0]-x["prediction"][0]) for x in natural]),result["without_engineered_controls"]["mae"]["overall"],places=6)
 def test_deployment_receipt(self):
  receipt=json.loads((ROOT/"run/deployment_verification.json").read_text())
  self.assertTrue(receipt["portable_predictor_matches_sealed_predictions"]);self.assertLess(receipt["max_absolute_difference"],3e-5)
if __name__=="__main__":unittest.main()
