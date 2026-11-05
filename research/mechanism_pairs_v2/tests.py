import unittest,json,sys
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import runpy
ns=runpy.run_path(str(R/"build.py"),run_name="mechanism_builder")
restore,certify,state_key=[ns[k] for k in ["restore","certify","state_key"]]
class Tests(unittest.TestCase):
 def test_all(self):
  rows={r["id"]:r for r in map(json.loads,(R/"dataset.jsonl").read_text().splitlines())};ev=json.loads((R/"mechanism_evidence.json").read_text())
  for r in rows.values():
   p=restore({"hands":[" ".join(str(c) if c<=10 else {11:"J",12:"Q",13:"K",14:"A",15:"2",16:"X",17:"D"}[c] for c in h) for h in r["input"]["hands"]],**{k:r["input"][k] for k in ["turn","last","passes"]}})
   s,v=certify(p);self.assertEqual(state_key(p),r["verification"]["key"]);self.assertFalse(any(p.winner_after(m) for m in p.legal()))
   for first,chain,want in [(ev[r["id"]]["good_first"],ev[r["id"]]["good_line"],True),(ev[r["id"]]["bad_action"],ev[r["id"]]["bad_line"],False)]:
    q=p
    if not want:
     for text in ev[r["id"]]["bad_prefix"]:q=q.play(next(m for m in q.legal() if m.text()==text))
    vv={m.text():w for m,w in s.move_values(q)};self.assertEqual(vv[first],want);self.assertTrue(any(vv.values()))
    side=q.side(q.turn);names=[first]+[x["move"] for x in chain]
    for i,name in enumerate(names):
     m=next(m for m in q.legal() if m.text()==name);winner=q.winner_after(m)
     if winner:self.assertEqual(i,len(names)-1)
     q=q.play(m)
    self.assertIsNotNone(winner);self.assertEqual(winner==side,want)
  for pair in json.loads((R/"pairs.json").read_text()):
   a,b=[rows[pair[k]] for k in ["hard_id","easy_id"]]
   self.assertEqual(Counter(sum(a["input"]["hands"],[])),Counter(sum(b["input"]["hands"],[])))
   self.assertEqual([len(h) for h in a["input"]["hands"]],[len(h) for h in b["input"]["hands"]])
   self.assertEqual(a["provenance"]["family"],b["provenance"]["family"])
   self.assertGreater(a["target"]["overall"],b["target"]["overall"])
if __name__=="__main__":unittest.main()
