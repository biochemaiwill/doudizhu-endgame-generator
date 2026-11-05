"""Search count- and rank-multiset-matched, nonterminal counterfactuals."""
from pathlib import Path
import sys,json,random,time,signal,hashlib
from collections import Counter
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
OUT=BASE/"research/expansion_v3_pairs"
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
def run():
 if (OUT/"ledger.jsonl").exists():raise SystemExit("refuse overwrite")
 rows=list(map(json.loads,(BASE/"research/scorer_v2/dataset.jsonl").read_text().splitlines()))
 byid={r["id"]:r for r in rows};seen={r["verification"]["key"] for r in rows}
 rng=random.Random(2026110514);start=time.monotonic();candidates=[];counts=Counter()
 parents=["EX208","EX220","EX203","EX211","EX222","EX225","DZ094","DZ098","DZ099"]
 with (OUT/"ledger.jsonl").open("w") as ledger:
  for pid in parents:
   parent=byid[pid];state=parent["input"]
   if state["last"] is not None:continue
   hands=state["hands"];swaps=[]
   for a in range(3):
    for b in range(a+1,3):
     for ca in sorted(set(hands[a])):
      for cb in sorted(set(hands[b])):
       if ca!=cb:swaps.append((a,b,ca,cb))
   rng.shuffle(swaps);found=0
   for j,(a,b,ca,cb) in enumerate(swaps):
    if time.monotonic()-start>300 or found>=3:break
    hs=[list(h) for h in hands];hs[a].remove(ca);hs[a].append(cb);hs[b].remove(cb);hs[b].append(ca)
    pos=Position(tuple(hand_from_cards(h) for h in hs),state["turn"])
    r={"parent_id":pid,"parent_grade":parent["target"]["overall"],"origin_id":f"{pid}-swap-{a}-{b}-{ca}-{cb}","family":parent["provenance"]["family"],"hands":[hand_to_text(h) for h in pos.hands],"turn":pos.turn,"last":None,"passes":0,"swap":[a,b,ca,cb],"source":"synthetic_counterfactual_same_counts_and_rank_multiset"}
    try:
     key=state_key(pos);r["key"]=key
     if key in seen:continue
     seen.add(key);validate_position(pos);signal.setitimer(signal.ITIMER_REAL,.6)
     s=Solver(node_limit=180000);vals=s.move_values(pos);wins=[m for m,w in vals if w]
     if any(pos.winner_after(m) for m,w in vals):r["status"]="reject_immediate_finish"
     elif not wins:r["status"]="losing"
     else:
      legal=[m for m,w in vals if not m.is_pass]
      largest=max(legal,key=lambda m:(m.n,-m.main))
      singles=[m for m in legal if m.n==1]
      smallest=min(singles,key=lambda m:m.main) if singles else largest
      if largest not in wins or smallest not in wins or len(wins)/len(vals)<.5:r["status"]="not_simpler_filter"
      else:
       signal.setitimer(signal.ITIMER_REAL,10)
       assert ref.move_values(pos)==[((t,c),w) for t,c,w in signature(vals)]
       assert signature(Solver(node_limit=2000000,canonical=False).move_values(pos))==signature(vals)
       r["moves"]=[move_record(m,w) for m,w in vals];r["winning_moves"]=[m.text() for m in wins]
       r["pv"]=trace(pos,s,maxplies=70)
       r["trap_examples"]=[{"first":m.text(),"continuation":trace(pos.play(m),s,maxplies=70)} for m,w in vals if not w][:2]
       r["status"]="verified_unlabeled";candidates.append(r);found+=1
    except (TimeBudget,BudgetExceeded) as e:r["status"]="unknown_budget"
    except Exception as e:r["status"]="error";r["error"]=repr(e)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    ledger.write(json.dumps(r,ensure_ascii=False)+"\n");ledger.flush();counts[r["status"]]+=1
   print(pid,"candidates",found,flush=True)
 (OUT/"candidates.json").write_text(json.dumps(candidates,ensure_ascii=False,indent=2))
 (OUT/"manifest.json").write_text(json.dumps({"seed":2026110514,"seconds":time.monotonic()-start,"counts":dict(counts),"verified_unlabeled":len(candidates),"filter_is_not_human_difficulty":True},indent=2))
 print("DONE",len(candidates),dict(counts),flush=True)
if __name__=="__main__":run()
