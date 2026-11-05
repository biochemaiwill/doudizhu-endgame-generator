import unittest,json
from pathlib import Path
import numpy as np
from data import load_rows,make_split,components,class_weights,support_ids
ROOT=Path(__file__).resolve().parent
class SplitTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=load_rows();cls.pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text());cls.split=make_split(cls.rows,cls.pairs)
 def test_no_family_url_or_pair_crosses_split(self):
  group=components(self.rows,self.pairs);owner={i:p for p,ids in self.split["ids"].items() for i in ids}
  for ids in group.values():self.assertEqual(len({owner[i] for i in ids}),1)
  self.assertEqual(owner["DZ095"],owner["EX238"])
  for p in self.pairs:self.assertEqual(owner[p["hard_id"]],owner[p["easy_id"]])
 def test_partition_all_grades_new_only_holdouts(self):
  ids=self.split["ids"];self.assertEqual(sum(map(len,ids.values())),153)
  self.assertEqual(len(set(sum(ids.values(),[]))),153)
  self.assertTrue(all(i.startswith("EX") for i in ids["val"]+ids["test"]))
  self.assertTrue(all(n>0 for part in self.split["counts"].values() for n in part.values()))
 def test_equal_total_grade_weights(self):
  y=np.array([1]*8+[2]*5+[3]*3+[4]*2);w=class_weights(y)
  for g in range(1,5):self.assertAlmostEqual(float(w[y==g].sum()),len(y)/4,places=5)
 def test_fewshot_train_only_unique(self):
  for seed in [101,202,303,404,505]:
   ids=support_ids(self.rows,self.split["ids"]["train"],seed)
   self.assertEqual(len(ids),8);self.assertEqual(len(set(ids)),8)
   self.assertTrue(set(ids)<=set(self.split["ids"]["train"]))
   labels={r["id"]:r["target"]["overall"] for r in self.rows}
   self.assertEqual([sum(labels[i]==g for i in ids) for g in range(1,5)],[2]*4)
 def test_split_determinism(self):
  self.assertEqual(self.split,make_split(self.rows,self.pairs))
if __name__=="__main__":unittest.main()
