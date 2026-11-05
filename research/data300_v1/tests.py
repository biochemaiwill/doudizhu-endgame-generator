from pathlib import Path
import json,hashlib,unittest
R=Path(__file__).resolve().parent
def jl(p):return [json.loads(s) for s in p.read_text().splitlines()]
class DataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=jl(R/"dataset_merged.jsonl");cls.byid={x["id"]:x for x in cls.rows}
 def test_count_unique_and_hash(self):
  m=json.loads((R/"manifest.json").read_text())
  self.assertEqual(len(self.rows),m["merged_rows"]);self.assertGreaterEqual(len(self.rows),300)
  self.assertEqual(len(self.rows),len(self.byid));self.assertEqual(len(self.rows),len({r["verification"]["key"] for r in self.rows}))
  self.assertEqual(hashlib.sha256((R/"dataset_merged.jsonl").read_bytes()).hexdigest(),m["sha256"])
 def test_old189_unchanged(self):
  old=jl(R.parent/"scorer_v4/dataset.jsonl")+jl(R.parent/"grade4_expansion_v5/dataset.jsonl")
  for r in old:
   self.assertEqual(r["input"],self.byid[r["id"]]["input"]);self.assertEqual(r["target"],self.byid[r["id"]]["target"])
 def test_conflicts_excluded(self):
  banned={x["key"] for x in json.loads((R/"quarantined_conflicts.json").read_text())}
  self.assertFalse(banned & {r["verification"]["key"] for r in self.rows})
 def test_labels_and_pairs(self):
  for r in self.rows:
   self.assertTrue(all(type(x)==int and 1<=x<=4 for x in r["target"].values()));self.assertFalse(r["annotation"]["human_calibrated"])
  for p in json.loads((R/"pairs.json").read_text()):
   a,b=self.byid[p["hard_id"]],self.byid[p["easy_id"]]
   self.assertEqual(a["provenance"]["family"],b["provenance"]["family"])
   self.assertGreater(a["target"]["overall"],b["target"]["overall"])
if __name__=="__main__":unittest.main()
