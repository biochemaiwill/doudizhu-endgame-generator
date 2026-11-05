"""Reproducible independent synthetic deals; search filters are not labels."""
from pathlib import Path
import sys,json,random,time,signal,platform,hashlib
from collections import Counter
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from endgame.fast import hand_from_text
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/"research/expansion_v2"
SEED=2026110507
def run():
 target=OUT/"search-ledger.jsonl"
 if target.exists() or target.with_suffix(target.suffix+".gz").exists():raise SystemExit("Refuse overwrite")
 rng=random.Random(SEED);start=time.monotonic();counts=Counter();accepted=[]
 seen={r["verification"]["key"] for r in map(json.loads,(BASE/"research/pilot100/dataset.jsonl").read_text().splitlines())}
 for f in [BASE/"research/highscore_candidates_v1/candidates.jsonl"]:
  for r in map(json.loads,f.read_text().splitlines()):
   k=r.get("verification",{}).get("key",r.get("key"))
   if k:seen.add(k)
 with target.open("w") as ledger:
  for i in range(4000):
   if time.monotonic()-start>600 or len(accepted)>=100:break
   ranks=sorted(rng.sample(range(3,16),rng.randint(5,9)))
   deck=[x for x in full_deck() if x in ranks or x>=16];rng.shuffle(deck)
   sizes=[rng.randint(4,8) for _ in range(3)]
   while sum(sizes)>21:sizes[rng.randrange(3)]-=1
   if sum(sizes)>len(deck):continue
   k=0;hands=[]
   for n in sizes:hands.append(hand_from_cards(deck[k:k+n]));k+=n
   pos=Position(tuple(hands),rng.randrange(3));history=[]
   if i%3==0:
    for _ in range(rng.randint(1,3)):
     ms=[m for m in pos.legal() if not pos.winner_after(m)]
     if not ms:break
     m=rng.choice(ms);history.append(dict(player=pos.turn,**move_record(m)));pos=pos.play(m)
   r=dict(origin_id=f"S{SEED}-{i:04d}",family=f"synthetic-deal-{SEED}-{i:04d}",source="independent_synthetic_remaining_hands",seed=SEED,attempt=i,history=history,
          hands=[hand_to_text(h) for h in pos.hands],turn=pos.turn,passes=pos.passes,last=move_record(pos.last) if pos.last else None,n_cards=sum(map(hand_size,pos.hands)))
   try:
    validate_position(pos);r["key"]=state_key(pos)
    if r["key"] in seen:r["status"]="duplicate"
    else:
     seen.add(r["key"]);signal.setitimer(signal.ITIMER_REAL,0.8)
     s=Solver(node_limit=180000);vals=s.move_values(pos);wins=[m for m,w in vals if w]
     r["moves"]=[move_record(m,w) for m,w in vals];r["winning_moves"]=[m.text() for m in wins]
     legal=[m for m,w in vals if not m.is_pass]
     maxm=max(legal,key=lambda m:(m.n,-m.main)) if legal else vals[0][0]
     single=[m for m in legal if m.n==1];low=min(single,key=lambda m:m.main) if single else maxm
     if not wins:r["status"]="losing"
     elif any(pos.winner_after(m) for m in wins):r["status"]="immediate_win"
     elif len(vals)<4 or len(wins)>2:r["status"]="not_narrow"
     elif maxm in wins and low in wins:r["status"]="simple_filters"
     else:
      signal.setitimer(signal.ITIMER_REAL,5)
      rv=ref.move_values(pos)
      assert rv==[((t,c),w) for t,c,w in signature(vals)]
      assert signature(vals)==signature(Solver(node_limit=500000,canonical=False).move_values(pos))
      r["reference_states"]=ref.solve.cache_info().currsize
      r["pv"]=trace(pos,s,maxplies=60)
      traps=sorted([m for m,w in vals if not w],key=lambda m:(m not in [maxm,low],-m.n,m.main))[:3]
      r["trap_examples"]=[dict(first=m.text(),continuation=trace(pos.play(m),s,maxplies=60)) for m in traps]
      r["status"]="verified";accepted.append(r)
   except (TimeBudget,BudgetExceeded) as e:r["status"]="unknown_budget";r["error"]=type(e).__name__
   except Exception as e:r["status"]="error";r["error"]=repr(e)
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   ledger.write(json.dumps(r,ensure_ascii=False)+"\n");ledger.flush();counts[r["status"]]+=1
   if i%100==0:print(dict(attempt=i,accepted=len(accepted),seconds=round(time.monotonic()-start),counts=dict(counts)),flush=True)
 (OUT/"verified-pool.json").write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
 (OUT/"run.json").write_text(json.dumps(dict(seed=SEED,counts=dict(counts),accepted=len(accepted),seconds=time.monotonic()-start,python=sys.version,platform=platform.platform(),max_attempts=4000,wall_budget_seconds=600,filters="narrow nontrivial; NOT difficulty labels"),indent=2))
 print("DONE",dict(counts),len(accepted),flush=True)
if __name__=="__main__":run()
