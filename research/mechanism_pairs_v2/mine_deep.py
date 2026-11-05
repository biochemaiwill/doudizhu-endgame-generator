from pathlib import Path
import sys,json,random,signal,time
B=Path("/home/wutianxiang/doudizhu-endgame-generator");O=B/"research/mechanism_pairs_v2";O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
pool=json.loads((B/"research/expansion_v3_synthetic/verified-pool.json").read_text())
blocked=set()
for f in [B/"research/scorer_v4/dataset.jsonl",B/"research/grade4_expansion_v5/dataset.jsonl",B/"research/mechanism_pairs_v1/dataset.jsonl"]:
 blocked.update(json.loads(s)["verification"]["key"] for s in f.read_text().splitlines())
rng=random.Random(2026110525);out=[];start=time.time()
for r in pool:
 if r["key"] in blocked:continue
 p=restore(r);v=Solver().move_values(p);w=[m for m,b in v if b]
 if not w or any(p.winner_after(m) for m in w):continue
 critical=sum(1 for z in r["pv"] if z["player"]==p.turn and z["legal"]>=3 and 0<z["winning"]<z["legal"])
 if critical<3:continue
 typ=None
 if all(m.n==1 and p.hands[p.turn] for m in w):
  cards=hand_to_cards(p.hands[p.turn])
  typ="拆对子" if any(cards.count(m.cards[0])>=2 for m in w) else "控制次序"
 if all(m.n>=4 for m in w):typ="带牌选择"
 if typ is None or sum(x["mechanism"]==typ for x in out)>=2:continue
 hs=[hand_to_cards(h) for h in p.hands];trials=[]
 for a in range(3):
  for b in range(a+1,3):
   for ca in set(hs[a]):
    for cb in set(hs[b]):
     if ca!=cb:trials.append((a,b,ca,cb))
 rng.shuffle(trials)
 for a,b,ca,cb in trials:
  if time.time()-start>180:break
  hh=[h[:] for h in hs];hh[a].remove(ca);hh[a].append(cb);hh[b].remove(cb);hh[b].append(ca)
  q=Position(tuple(hand_from_cards(h) for h in hh),p.turn,p.last,p.passes)
  if state_key(q) in blocked:continue
  try:
   signal.setitimer(signal.ITIMER_REAL,2)
   sv=Solver(node_limit=500000);vv=sv.move_values(q);ww=[m for m,z in vv if z]
   if len(ww)<len(vv)/2 or any(q.winner_after(m) for m in ww):continue
   epv=trace(q,sv,maxplies=70)
   ec=sum(1 for z in epv if z["player"]==q.turn and z["legal"]>=3 and 0<z["winning"]<z["legal"])
   if ec>1:continue
   assert ref.move_values(q)==[((t,c),z) for t,c,z in signature(vv)]
   assert signature(vv)==signature(Solver(node_limit=500000,canonical=False).move_values(q))
   out.append(dict(mechanism=typ,hard=r,easy=dict(hands=[hand_to_text(h) for h in q.hands],turn=q.turn,last=r.get("last"),passes=q.passes,key=state_key(q),moves=[move_record(m,z) for m,z in vv],winning_moves=[m.text() for m in ww]),swap=[a,b,ca,cb]))
   print(typ,r["origin_id"],r["hands"],r["winning_moves"],"EASY",out[-1]["easy"]["hands"],out[-1]["easy"]["winning_moves"],flush=True);break
  except (TimeBudget,BudgetExceeded):pass
  finally:signal.setitimer(signal.ITIMER_REAL,0)
 if len(out)>=6 or time.time()-start>180:break
(O/"candidate_pairs_deep.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print("DONE",len(out))
