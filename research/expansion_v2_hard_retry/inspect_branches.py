"""Regenerate illustrative branches with the unchanged vendored solver."""
from pathlib import Path
import sys,json,signal
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
d=Path(__file__).resolve().parent
initial=restore(json.loads((d/"input.json").read_text()))
s=Solver(node_limit=20000000)
signal.setitimer(signal.ITIMER_REAL,90)
def at(seq):
 p=initial
 for text in seq:
  m=next(m for m,w in s.move_values(p) if m.text()==text)
  p=p.play(m)
 return p
p=at(["2"])
out=[]
for a,w in s.move_values(p):
 p1=p.play(a)
 out.append({"farmer_down":a.text(),"outcome_for_farmer":w,"farmer_up":[{"move":b.text(),"wins":v} for b,v in s.move_values(p1)]})
(d/"after_two.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
vals=s.move_values(initial)
out={"warning":"Lines illustrate one legal continuation; losing side moves are arbitrary, not human-optimal defense.","winning_line":trace(initial,s,maxplies=70),"traps":[]}
for m,w in vals:
 if not w and (m.n==10 or m.text()=="6" or m.text()=="6 6"):
  out["traps"].append({"first":m.text(),"line":trace(initial.play(m),s,maxplies=70)})
(d/"illustrative_lines.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
p=at(["2","大王","过","过"])
out={"prefix":["landlord:2","farmer_down:big_joker","farmer_up:pass","landlord:pass"],"hands":[hand_to_text(h) for h in p.hands],"turn":p.turn,"branches":[]}
for a,w in s.move_values(p):
 p1=p.play(a)
 for b,v in s.move_values(p1):
  if p1.winner_after(b):continue
  p2=p1.play(b);vs=s.move_values(p2)
  out["branches"].append({"farmer_down_lead":a.text(),"farmer_up_response":b.text(),"landlord_wins":[m.text() for m,z in vs if z],"landlord_loses":[m.text() for m,z in vs if not z]})
(d/"after_joker_defense.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
out=[]
for seq in [["2","过","过"],["2","大王","过","过","7","过","Q"],["2","大王","过","过","9","过","Q"],["2","大王","过","过","7","过","Q","A","过","过"]]:
 p=at(seq);vs=s.move_values(p)
 out.append({"prefix":seq,"hands":[hand_to_text(h) for h in p.hands],"turn":p.turn,"last":p.last.text() if p.last else None,"winning":[m.text() for m,w in vs if w],"all":[move_record(m,w) for m,w in vs],"line":trace(p,s,maxplies=60)})
(d/"key_followups.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
