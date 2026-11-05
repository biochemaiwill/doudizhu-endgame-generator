import json,sys,unittest,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"pilot100"))
from build import Position,hand_from_cards,state_key,Solver,signature
import reference_ordered as ref
class MechanismTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows={r["id"]:r for r in map(json.loads,(ROOT/"dataset.jsonl").read_text().splitlines())}
  cls.ev=json.loads((ROOT/"mechanism_evidence.json").read_text())
  cls.pairs=json.loads((ROOT/"pairs.json").read_text())
 def position(self,ident):
  x=self.rows[ident]["input"];return Position(tuple(hand_from_cards(h) for h in x["hands"]),x["turn"])
 def replay(self,p,names,require_terminal=False):
  winner=None
  for i,name in enumerate(names):
   m=next(m for m in p.legal() if m.text()==name)
   winner=p.winner_after(m)
   if winner:self.assertEqual(i,len(names)-1)
   p=p.play(m)
  if require_terminal:self.assertIsNotNone(winner)
  return p,winner
 def test_contract_and_reuse(self):
  self.assertEqual(len(self.rows),8);self.assertEqual(len(self.pairs),4)
  self.assertEqual(sum(not r["pilot_provenance"]["reused_existing"] for r in self.rows.values()),1)
  self.assertEqual(len({r["verification"]["key"] for r in self.rows.values()}),8)
  for r in self.rows.values():
   self.assertFalse(r["annotation"]["human_calibrated"])
   self.assertEqual(r["annotation"]["rubric"],"v0.1")
 def test_pair_conservation(self):
  for x in self.pairs:
   a,b=self.rows[x["hard_id"]],self.rows[x["easy_id"]]
   self.assertGreater(a["target"]["overall"],b["target"]["overall"])
   self.assertEqual(a["provenance"]["family"],b["provenance"]["family"])
   self.assertEqual([len(h) for h in a["input"]["hands"]],[len(h) for h in b["input"]["hands"]])
   self.assertEqual(Counter(sum(a["input"]["hands"],[])),Counter(sum(b["input"]["hands"],[])))
   for i in [a["id"],b["id"]]:self.assertFalse(any(self.position(i).winner_after(m) for m in self.position(i).legal()))
 def test_root_all_actions_three_way(self):
  for ident,r in self.rows.items():
   p=self.position(ident);vals=Solver(node_limit=2000000).move_values(p);sig=signature(vals)
   self.assertEqual(sig,signature(Solver(node_limit=2000000,canonical=False).move_values(p)))
   self.assertEqual([((t,c),w) for t,c,w in sig],ref.move_values(p))
   self.assertEqual(state_key(p),r["verification"]["key"])
 def test_mechanism_conditional_claims(self):
  for claim in self.ev["mechanism_facts"]:
   p,_=self.replay(self.position(claim["hard_id"]),claim["prefix_text"])
   vals={m.text():w for m,w in Solver().move_values(p)}
   self.assertEqual(sorted(k for k,v in vals.items() if v),sorted(claim["exact_winning_actions"]))
   for name in claim["checked_losing_actions"]:self.assertFalse(vals[name])
   for key,wanted in [("correct",True),("wrong",False)]:
    q,winner=self.replay(p,[claim[key+"_action"]]+[s["move"] for s in claim[key+"_continuation"]],True)
    self.assertEqual(winner==p.side(p.turn),wanted)
 def test_all_position_wrong_and_correct_lines(self):
  for ident,e in self.ev["positions"].items():
   p=self.position(ident);side=p.side(p.turn)
   _,winner=self.replay(p,[e["example_winning_first"]]+[x["move"] for x in e["winning_continuation"]],True)
   self.assertEqual(winner,side)
   q,_=self.replay(p,e["wrong_prefix"])
   self.assertEqual(q.side(q.turn),side)
   qvalues={m.text():w for m,w in Solver().move_values(q)}
   self.assertTrue(any(qvalues.values()))
   self.assertFalse(qvalues[e["example_wrong_action"]])
   _,winner=self.replay(q,[e["example_wrong_action"]]+[x["move"] for x in e["wrong_continuation"]],True)
   self.assertNotEqual(winner,side)
if __name__=="__main__":unittest.main()
