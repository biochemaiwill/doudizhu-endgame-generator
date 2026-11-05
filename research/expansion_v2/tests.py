"""Contract checks for the isolated reviewed expansion."""
import unittest,json,hashlib
from pathlib import Path
from assemble import restore,validate_position,state_key,hand_size,hand_to_cards,Position,read_blocked
ROOT=Path(__file__).resolve().parent
def readlines(p):return [json.loads(x) for x in p.read_text().splitlines()]
class DataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=readlines(ROOT/"dataset.jsonl");cls.evidence=json.loads((ROOT/"evidence.json").read_text())
  cls.byid={r["id"]:r for r in cls.rows};cls.ev={r["id"]:r for r in cls.evidence}
 def test_original_frozen_data_unchanged(self):
  p=ROOT.parent/"pilot100/dataset.jsonl"
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),"85b3d83ca0ed203801161299c83178f858b8e0d3e3a6a288341ca759ada0b59c")
 def test_unique_and_disjoint_from_prior(self):
  self.assertEqual(len(self.rows),len(self.byid))
  keys=[r["verification"]["key"] for r in self.rows]
  self.assertEqual(len(keys),len(set(keys)));self.assertFalse(set(keys)&read_blocked())
 def test_serialized_inputs_match_verified_positions(self):
  for r in self.rows:
   e=self.ev[r["id"]];p=restore(e);validate_position(p)
   self.assertEqual(r["input"]["hands"],[hand_to_cards(h) for h in p.hands])
   self.assertEqual(r["input"]["turn"],p.turn);self.assertEqual(r["input"]["passes"],p.passes)
   self.assertEqual(r["input"]["last"],dict(cards=list(p.last.cards),type=int(p.last.type),main=p.last.main,length=p.last.length) if p.last else None)
   self.assertEqual(r["verification"]["key"],state_key(p))
   legal={(int(m.type),tuple(m.cards)) for m in p.legal()}
   self.assertEqual(legal,{(m["type"],tuple(m["cards"])) for m in e["moves"]})
   self.assertEqual(e["fresh_verification"]["three_way_root_outcomes"],"pass")
   self.assertTrue(e["winning_moves"])
 def test_labels_and_caveats(self):
  for r in self.rows:
   self.assertTrue(all(type(v)==int and 1<=v<=5 for v in r["target"].values()))
   self.assertTrue(r["reason"]);self.assertFalse(r["annotation"]["human_calibrated"])
   self.assertEqual(r["annotation"]["rubric"],"v0.1")
 def test_pairs_exact_counts_role_and_phase(self):
  pairs=json.loads((ROOT/"pairs.json").read_text());self.assertGreaterEqual(len(pairs),5)
  for p in pairs:
   hard=self.byid[p["hard_id"]];easy=self.byid[p["easy_id"]]
   self.assertEqual([len(x) for x in hard["input"]["hands"]],[len(x) for x in easy["input"]["hands"]])
   for k in ["turn","last","passes"]:self.assertEqual(hard["input"][k],easy["input"][k])
   self.assertGreater(hard["target"]["overall"],easy["target"]["overall"])
   self.assertEqual(hard["provenance"]["family"],easy["provenance"]["family"])
   pos=restore(self.ev[easy["id"]])
   self.assertTrue(any(pos.winner_after(m) for m in pos.legal()))
 def test_hash_manifest_and_no_training_changes(self):
  manifest=json.loads((ROOT/"manifest.json").read_text())
  self.assertEqual(manifest["rows"],len(self.rows));self.assertEqual(manifest["three_way_verified"],len(self.rows))
  self.assertEqual(manifest["dataset_sha256"],hashlib.sha256((ROOT/"dataset.jsonl").read_bytes()).hexdigest())
  self.assertFalse(manifest["training_changed"])
if __name__=="__main__":unittest.main()
