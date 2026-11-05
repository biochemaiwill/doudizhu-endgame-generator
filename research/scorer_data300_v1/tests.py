import unittest,json
from pathlib import Path
import numpy as np
from compare import load,CONFIGS
R=Path(__file__).resolve().parent
class ComparisonTests(unittest.TestCase):
 def test_split_preserves_old_validation_and_train(self):
  rows,lookup,folds,groups=load()
  old=json.loads((R.parent/"scorer_v6/cv_split.json").read_text())["folds"]
  owner={i:g for g,ids in groups.items() for i in ids}
  for f,o in zip(folds,old):
   self.assertEqual(f["val_ids"],o["val_ids"]);self.assertEqual(f["old_train_ids"],o["train_ids"])
   self.assertTrue(set(f["old_train_ids"])<=set(f["expanded_train_ids"]))
   self.assertFalse({owner[i] for i in f["val_ids"]}&{owner[i] for i in f["expanded_train_ids"]})
 def test_saved_metrics_and_probabilities(self):
  results=json.loads((R/"results.json").read_text());pred=json.loads((R/"oof_predictions.json").read_text())
  for name,arms in pred.items():
   for arm,records in arms.items():
    scores=[]
    for seed in [17,73]:
     for fold in range(3):
      rr=[r for r in records if r["seed"]==seed and r["fold"]==fold]
      p=np.array([r["probabilities"] for r in rr]);y=np.array([r["truth"] for r in rr])
      self.assertEqual(p.shape[1:],(3,4));np.testing.assert_allclose(p.sum(-1),1,atol=1e-5)
      scores.append(float(np.mean(p[:,0].argmax(-1)==y[:,0])))
    self.assertAlmostEqual(np.mean(scores),results[name][arm]["mean_accuracy"],places=7)
 def test_old_reproduces_frozen_v6(self):
  results=json.loads((R/"results.json").read_text());old=json.loads((R.parent/"scorer_v6/cv_results.json").read_text())
  for name,cfg in CONFIGS.items():
   key="_".join(f"{k}-{v}" for k,v in cfg.items())
   self.assertAlmostEqual(results[name]["old"]["mean_accuracy"],old[key]["mean_accuracy"],places=6)
if __name__=="__main__":unittest.main()
