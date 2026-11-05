"""Bounded 507273 certification retry; does not alter rules."""
from pathlib import Path
import sys,json,time,signal,resource,platform
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/"research/expansion_v2_hard_retry"
resource.setrlimit(resource.RLIMIT_CPU,(295,300))
data=json.loads((BASE/"research/pilot100/public-inputs.json").read_text())
v=next(r for r in data if r[0]=="33iq-507273")
row=dict(source_id=v[0],hands=v[1],turn=v[2],last_text=v[3],passes=v[4],url=v[5],note=v[6])
(OUT/"input.json").write_text(json.dumps(row,ensure_ascii=False,indent=2))
p=restore(row);validate_position(p)
result=dict(input=row,key=state_key(p),python=sys.version,platform=platform.platform(),modes=[],status="pending",overall=None,note="Search effort is not human difficulty; no score unless certified and reviewed.")
def save():
 (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
save()
sigs=[]
for label in ("canonical","uncanonical","independent"):
 start=time.process_time()
 try:
  signal.setitimer(signal.ITIMER_REAL,90 if label!="independent" else 120)
  if label=="independent":
   vals=ref.move_values(p);sig=[(a,b,c) for ((a,b),c) in vals]
   detail=dict(states=ref.solve.cache_info().currsize,root_signature=sig)
  else:
   s=Solver(node_limit=20000000,canonical=label=="canonical");vals=s.move_values(p)
   sig=signature(vals);detail=dict(winning_moves=[m.text() for m,w in vals if w],moves=[move_record(m,w) for m,w in vals],root_signature=sig)
   if label=="canonical":
    result["winning_moves"]=detail["winning_moves"]
  sigs.append(sig);result["modes"].append(dict(mode=label,status="completed",cpu_seconds=time.process_time()-start,**detail))
 except (TimeBudget,BudgetExceeded) as e:result["modes"].append(dict(mode=label,status="unknown_budget",error=type(e).__name__,cpu_seconds=time.process_time()-start))
 except Exception as e:result["modes"].append(dict(mode=label,status="error",error=repr(e),cpu_seconds=time.process_time()-start))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 save();print(label,result["modes"][-1]["status"],result["modes"][-1]["cpu_seconds"],flush=True)
result["status"]="verified" if len(sigs)==3 and all(s==sigs[0] for s in sigs) else "unknown_or_mismatch"
save();print("FINAL",result["status"],result.get("winning_moves"),flush=True)
