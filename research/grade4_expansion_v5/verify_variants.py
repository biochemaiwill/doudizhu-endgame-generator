from pathlib import Path
import sys,json,signal,time
B=Path("/home/wutianxiang/doudizhu-endgame-generator");O=B/"research/grade4_expansion_v5";O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
rows=list(map(json.loads,(B/"research/scorer_v4/dataset.jsonl").read_text().splitlines()))
by={r["id"]:r for r in rows};seen={r["verification"]["key"] for r in rows}
for p in [B/"research/highscore_candidates_v1/candidates.jsonl",B/"research/expansion_v3_synthetic/formal_candidates.jsonl"]:
 for r in map(json.loads,p.read_text().splitlines()):
  k=r.get("key",r.get("verification",{}).get("key"))
  if k:seen.add(k)
out=[]
specs=[("EX208",1,4,3),("EX208",1,5,6),("EX208",2,4,3),("EX220",0,8,6),("EX220",0,8,10),("EX220",1,4,3),("EX220",2,4,3),("DZ098",2,3,6),("DZ098",2,4,7),("DZ099",0,4,5),("DZ099",0,5,4),("DZ094",1,5,3)]
for pid,seat,old,new in specs:
 a=by[pid];h=[list(x) for x in a["input"]["hands"]]
 if old not in h[seat]:continue
 h[seat].remove(old);h[seat].append(new)
 r=dict(origin_id=f"{pid}-s{seat}-{old}to{new}",parent_id=pid,family=a["provenance"]["family"],source="synthetic_variant",hands=[hand_to_text(hand_from_cards(x)) for x in h],turn=a["input"]["turn"],passes=0,last=None,mutation=[seat,old,new])
 try:
  p=restore(r);validate_position(p);r["key"]=state_key(p)
  if r["key"] in seen:r["status"]="duplicate"
  else:
   seen.add(r["key"]);signal.setitimer(signal.ITIMER_REAL,25);solver=Solver(node_limit=2500000);v=solver.move_values(p)
   assert ref.move_values(p)==[((t,c),w) for t,c,w in signature(v)]
   assert signature(v)==signature(Solver(node_limit=2500000,canonical=False).move_values(p))
   r.update(status="verified",moves=[move_record(m,w) for m,w in v],winning_moves=[m.text() for m,w in v if w],pv=trace(p,solver,maxplies=70),trap_examples=[dict(first=m.text(),continuation=trace(p.play(m),solver,maxplies=70)) for m,w in v if not w][:3])
 except (TimeBudget,BudgetExceeded):r["status"]="unknown_budget"
 except Exception as e:r.update(status="invalid_or_error",error=repr(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 out.append(r);(O/"candidate_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2));print(r["origin_id"],r["status"],r.get("winning_moves"),flush=True)
