import json,unittest,hashlib,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"pilot100"))
from build import validate_position,state_key
from verify_sources import restore
class ExpansionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.new=[json.loads(s) for s in (ROOT/"dataset.jsonl").read_text().splitlines()]
  cls.old=[json.loads(s) for s in (ROOT.parent/"scorer_v2/dataset.jsonl").read_text().splitlines()]
  cls.all={r["id"]:r for r in cls.old+cls.new}
  cls.ev={r["id"]:r for r in json.loads((ROOT/"evidence.json").read_text())}
 def test_uniqueness_and_old_frozen(self):
  self.assertEqual(len({r["verification"]["key"] for r in self.old+self.new}),len(self.old)+len(self.new))
  m=json.loads((ROOT.parent/"scorer_v2/run/environment.json").read_text())
  self.assertEqual(hashlib.sha256((ROOT.parent/"scorer_v2/dataset.jsonl").read_bytes()).hexdigest(),m["hashes"]["dataset.jsonl"])
 def test_evidence_and_targets(self):
  for r in self.new:
   p=restore(self.ev[r["id"]]["source_record"]);validate_position(p)
   self.assertEqual(state_key(p),r["verification"]["key"])
   self.assertFalse(r["annotation"]["human_calibrated"]);self.assertEqual(r["annotation"]["rubric"],"v0.1")
   self.assertTrue(all(type(x)==int and 1<=x<=5 for x in r["target"].values()))
   legal={(int(m.type),tuple(m.cards)) for m in p.legal()}
   ev=self.ev[r["id"]]["verified_moves"]
   self.assertEqual(legal,{(m["type"],tuple(m["cards"])) for m in ev})
   self.assertEqual(sorted(m["move"] for m in ev if m["mover_wins"]),sorted(r["verification"]["winning_moves"]))
 def test_pair_conservation_and_nonterminal(self):
  pairs=json.loads((ROOT/"pairs.json").read_text());self.assertGreaterEqual(len(pairs),5)
  for p in pairs:
   a,b=self.all[p["hard_id"]],self.all[p["easy_id"]];aa=a["input"];bb=b["input"]
   self.assertEqual([len(h) for h in aa["hands"]],[len(h) for h in bb["hands"]])
   self.assertEqual(Counter(sum(aa["hands"],[])),Counter(sum(bb["hands"],[])))
   self.assertEqual(a["provenance"]["family"],b["provenance"]["family"])
   for key in ["turn","last","passes"]:self.assertEqual(aa[key],bb[key])
   self.assertGreater(a["target"]["overall"],b["target"]["overall"])
   # Every new side has full source evidence; old parents have frozen proofs.
   for r in [a,b]:
    if r["id"] in self.ev:
     state=restore(self.ev[r["id"]]["source_record"])
     self.assertFalse(any(state.winner_after(m) for m in state.legal()))
 def test_manifest_hash(self):
  m=json.loads((ROOT/"manifest.json").read_text())
  self.assertEqual(len(self.new),m["rows"])
  self.assertEqual(hashlib.sha256((ROOT/"dataset.jsonl").read_bytes()).hexdigest(),m["sha256"])
  self.assertFalse(m["entered_old_training"])
if __name__=="__main__":unittest.main()
