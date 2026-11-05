"""Verify extracted source states with unchanged vendored rules. No source answers trusted."""
import sys,json,signal,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
ROOT=Path(__file__).resolve().parent
def verify(row):
 r=dict(row);started=time.monotonic()
 try:
  p=restore(row);validate_position(p);signal.setitimer(signal.ITIMER_REAL,60)
  s=Solver(node_limit=4000000);vals=s.move_values(p);sig=signature(vals)
  r["moves"]=[move_record(m,w) for m,w in vals]
  r["winning_moves"]=[m.text() for m,w in vals if w]
  assert ref.move_values(p)==[((t,c),w) for t,c,w in sig]
  assert signature(Solver(node_limit=4000000,canonical=False).move_values(p))==sig
  r["status"]="verified" if r["winning_moves"] else "losing"
  r["key"]=state_key(p);r["pv"]=trace(p,s,maxplies=60)
  r["trap_examples"]=[{"first":m.text(),"continuation":trace(p.play(m),s,maxplies=60)} for m,w in vals if not w][:3]
 except (TimeBudget,BudgetExceeded) as e:r["status"]="unknown_budget";r["error"]=repr(e)
 except Exception as e:r["status"]="error";r["error"]=repr(e)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r["elapsed_seconds"]=time.monotonic()-started
 return r
def at_path(p,path):
 for text in path:
  matches=[m for m in p.legal() if m.text()==text]
  assert len(matches)==1,(path,text)
  p=p.play(matches[0])
 return p
def branches(rows):
 out=[]
 for sid,path in [
 ("sina-jj20121107-after-Q",["2","过","小王","过","过"]),
 ("53shop-20230822-landlord",["9","过"])]:
  p=at_path(restore(next(r for r in rows if r["source_id"]==sid)),path)
  s=Solver(node_limit=4000000)
  for m in p.legal():
   q=p.play(m)
   vals=s.move_values(q)
   item={"source_id":sid,"path":path+[m.text()],"turn":q.turn,"hands":[hand_to_text(h) for h in q.hands],"moves":[move_record(a,w) for a,w in vals]}
   if sid.startswith("sina"):
    z=at_path(q,["过"])
    item["up_after_down_pass"]=[move_record(a,w) for a,w in s.move_values(z)]
   out.append(item)
 return out
if __name__=="__main__":
 rows=json.loads((ROOT/"inputs.json").read_text())
 results=[]
 for row in rows:
  rec=verify(row);results.append(rec)
  (ROOT/"verification.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
  print(rec["source_id"],rec["status"],rec.get("winning_moves"),flush=True)
 (ROOT/"critical-branches.json").write_text(json.dumps(branches(rows),ensure_ascii=False,indent=2))
 (ROOT/"manifest.json").write_text(json.dumps({"python":platform.python_version(),"rules":"unchanged pilot-v1 vendored bdee2f6e","command":"python3 research/expansion_v2_sources/verify.py","node_limit":4000000,"seconds_per_state":60,"outcome_checks":["optimized","canonical_off","independent_ordered"],"source_answers_trusted":False,"status_counts":dict(Counter(r["status"] for r in results))},indent=2))
