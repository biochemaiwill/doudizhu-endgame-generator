import sys,json,unittest
from pathlib import Path
import numpy as np
from scorer import encode_state, split_rows, OrdinalMLP
ROOT=Path(__file__).resolve().parent
ROWS=[json.loads(s) for s in (ROOT.parent/"pilot100/dataset.jsonl").read_text().splitlines()]
class ScorerTests(unittest.TestCase):
 def test_hand_order_invariant(self):
  a=ROWS[0]["input"];b=dict(a,hands=[list(reversed(h)) for h in a["hands"]])
  np.testing.assert_array_equal(encode_state(a),encode_state(b))
 def test_player_roles_not_exchangeable(self):
  a=ROWS[0]["input"];b=dict(a,hands=[a["hands"][1],a["hands"][0],a["hands"][2]])
  self.assertFalse(np.array_equal(encode_state(a),encode_state(b)))
 def test_targets_do_not_enter_features(self):
  a=dict(ROWS[0]["input"],target={"overall":5},reason="arbitrary",source="holdout")
  np.testing.assert_array_equal(encode_state(a),encode_state(ROWS[0]["input"]))
 def test_invalid_duplicate_joker_rejected(self):
  a=dict(ROWS[0]["input"],hands=[[17],[17],[3]])
  with self.assertRaises(ValueError):encode_state(a)
 def test_inconsistent_last_type_rejected(self):
  a=dict(ROWS[0]["input"],last={"cards":[3],"type":2,"main":3,"length":1},passes=0)
  with self.assertRaises(ValueError):encode_state(a)
 def test_split_counts_and_group_isolation(self):
  parts=split_rows(ROWS,20261105)
  self.assertEqual([len(parts[k]) for k in ("train","val","test")],[70,15,15])
  ids=[x for v in parts.values() for x in v]
  self.assertEqual(len(set(ids)),100)
  groups={k:{r["provenance"]["family"] for r in ROWS if r["id"] in parts[k]} for k in parts}
  self.assertFalse(groups["train"]&groups["val"]);self.assertFalse(groups["train"]&groups["test"]);self.assertFalse(groups["val"]&groups["test"])
  self.assertTrue(all(r["id"] in parts["train"] for r in ROWS if r["provenance"]["source"]=="MasterMind train API"))
  for k,target in [("train",[30,27,11,2]),("val",[6,6,2,1]),("test",[6,6,2,1])]:
   self.assertEqual([sum(r["target"]["overall"]==g for r in ROWS if r["id"] in parts[k]) for g in range(1,5)],target)
 def test_ordinal_probabilities_monotone(self):
  import torch
  m=OrdinalMLP(86);out=m(torch.zeros(4,86))
  self.assertEqual(tuple(out.shape),(4,3,3))
  self.assertTrue(torch.all(out[:,:,1:]<=out[:,:,:-1]).item())
  self.assertLess(sum(p.numel() for p in m.parameters()),10000)


class ArtifactTests(unittest.TestCase):
 def test_saved_model_matches_all_recorded_predictions(self):
  from scorer import predict_export
  run=ROOT/"runs/frozen100_20261105"
  saved=json.loads((run/"model.json").read_text())
  pred=predict_export(saved["artifact"],[r["input"] for r in ROWS])
  recorded={r["id"]:r["prediction"] for r in json.loads((run/"predictions.json").read_text())}
  np.testing.assert_allclose(pred,np.array([recorded[r["id"]] for r in ROWS]),atol=2e-5)
  self.assertTrue(np.all(pred>=1)); self.assertTrue(np.all(pred<=4))
  self.assertEqual(saved["unsupported_grades"],[5])
 def test_scaler_fit_on_training_rows_only(self):
  from scorer import encode_state,card_features
  run=ROOT/"runs/frozen100_20261105"
  parts=json.loads((run/"split.json").read_text())["ids"]
  registry=json.loads((run/"models.json").read_text())
  x=np.stack([encode_state(r["input"]) for r in ROWS if r["id"] in parts["train"]])
  for m in registry.values():
   if m["kind"]=="constant":continue
   z=card_features(x) if m.get("features")=="counts" else x
   np.testing.assert_allclose(np.array(m["mean"]),z.mean(0),atol=1e-6)

if __name__=="__main__":unittest.main()
