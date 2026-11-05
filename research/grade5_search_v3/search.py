"""Source-derived legal-prefix exploration; no labels inferred from search."""
from pathlib import Path
import sys,json,time,random,signal,hashlib,platform
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/"research/grade5_search_v3"
SRC=json.loads((OUT/"source-inputs.json").read_text())
rng=random.Random(2026110513)
ledger=[];valid=[];pool=[];t0=time.monotonic()
def save(): (OUT/"search.json").write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
for row in SRC:
 r=dict(row);r["kind"]="public_initial"
 try:
  p=restore(r);validate_position(p);r["counts"]=[hand_size(h) for h in p.hands];r["key"]=state_key(p);valid.append((r,p))
  signal.setitimer(signal.ITIMER_REAL,15)
  s=Solver(node_limit=1500000);v=s.move_values(p)
  sig=signature(v);r["optimized_winning_moves"]=[m.text() for m,w in v if w]
  r["status"]="optimized_only_not_certified"
 except (TimeBudget,BudgetExceeded) as e:r["status"]="unknown_budget";r["error"]=type(e).__name__
 except Exception as e:r["status"]="quarantine";r["error"]=repr(e)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 ledger.append(r);save();print(r["source_id"],r["status"],r.get("counts"),flush=True)
seen=set()
for k in range(120):
 if time.monotonic()-t0>210 or len(pool)>=8:break
 row,start=valid[k%len(valid)];p=start;hist=[]
 for ply in range(150):
  if sum(map(hand_size,p.hands))<=rng.randint(17,22):break
  choices=[m for m in p.legal() if not p.winner_after(m)]
  if not choices:break
  weights=[1+max(0,m.n-1)*2 for m in choices]
  m=rng.choices(choices,weights=weights)[0];hist.append(dict(player=p.turn,**move_record(m)));p=p.play(m)
 if sum(map(hand_size,p.hands))>22:continue
 r=dict(source_id=f'{row["source_id"]}-prefix-{k}',family=row["source_id"],kind="synthetic_legal_prefix_from_public_cards",source_url=row["source_url"],history=hist,hands=[hand_to_text(h) for h in p.hands],turn=p.turn,passes=p.passes,last=move_record(p.last) if p.last else None,key=state_key(p))
 if r["key"] in seen:continue
 seen.add(r["key"])
 try:
  validate_position(p);signal.setitimer(signal.ITIMER_REAL,1.5)
  s=Solver(node_limit=180000);vals=s.move_values(p);wins=[m for m,w in vals if w]
  r["moves"]=[move_record(m,w) for m,w in vals];r["winning_moves"]=[m.text() for m in wins]
  if not wins:r["status"]="losing"
  elif len(vals)<4 or len(wins)>2 or any(p.winner_after(m) for m in wins):r["status"]="not_narrow"
  else:
   signal.setitimer(signal.ITIMER_REAL,6)
   assert ref.move_values(p)==[((a,b),w) for a,b,w in signature(vals)]
   assert signature(Solver(node_limit=500000,canonical=False).move_values(p))==signature(vals)
   r["status"]="verified_unlabeled";r["pv"]=trace(p,s,60);r["trap_examples"]=[dict(first=m.text(),continuation=trace(p.play(m),s,60)) for m,w in vals if not w][:3];pool.append(r)
 except (TimeBudget,BudgetExceeded) as e:r["status"]="unknown_budget";r["error"]=type(e).__name__
 except Exception as e:r["status"]="error";r["error"]=repr(e)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 ledger.append(r);save()
 if k%10==0:print(k,len(pool),round(time.monotonic()-t0),flush=True)
(OUT/"verified-candidates.json").write_text(json.dumps(pool,ensure_ascii=False,indent=2))
(OUT/"manifest.json").write_text(json.dumps(dict(seed=2026110513,elapsed=time.monotonic()-t0,python=sys.version,platform=platform.platform(),status_counts=dict(Counter(r["status"] for r in ledger)),grade5_confirmed=0,labels_status="None inferred; requires human-style rubric review",command="python3 research/grade5_search_v3/search.py"),indent=2))
print("DONE",len(pool),dict(Counter(r["status"] for r in ledger)),flush=True)
