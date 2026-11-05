"""Assemble reviewed expansion and exact-count controls. Does not train models."""
import sys,json,random,signal,time,hashlib,platform,gzip
from pathlib import Path
from collections import Counter
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
OUT=BASE/"research/expansion_v2"
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from endgame.fast import hand_from_text
from verify_sources import restore
import reference_ordered as ref
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def validate_three(row):
 p=restore(row);validate_position(p);signal.setitimer(signal.ITIMER_REAL,150)
 start=time.monotonic()
 try:
  s=Solver(node_limit=30000000);vals=s.move_values(p)
  assert signature(vals)==signature(Solver(node_limit=30000000,canonical=False).move_values(p))
  assert [((t,c),w) for t,c,w in signature(vals)]==ref.move_values(p)
  assert any(w for m,w in vals),"losing row cannot enter"
  row.update(key=state_key(p),moves=[move_record(m,w) for m,w in vals],winning_moves=[m.text() for m,w in vals if w],status="verified")
  if "pv" not in row:row["pv"]=trace(p,s,maxplies=90)
  row["fresh_verification"]={"three_way_root_outcomes":"pass","seconds":time.monotonic()-start,"reference_states":ref.solve.cache_info().currsize}
  return p
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def read_blocked():
 rows=list(map(json.loads,(BASE/"research/pilot100/dataset.jsonl").read_text().splitlines()))
 rows+=list(map(json.loads,(BASE/"research/highscore_candidates_v1/candidates.jsonl").read_text().splitlines()))
 out={r.get("verification",{}).get("key",r.get("key")) for r in rows}
 for a in json.loads((BASE/"research/pilot-v1/verification.json").read_text())["cases"]:
  out.add(state_key(Position(tuple(hand_from_text(h) for h in a["hands"]),a["turn"])))
 return out
def main():
 if (OUT/"dataset.jsonl").exists():raise SystemExit("refuse overwrite")
 pool={r["origin_id"]:r for r in json.loads((OUT/"verified-pool.json").read_text())}
 labels=json.loads((OUT/"labels_agent.json").read_text())
 rows=[]
 for lab in labels:
  r=dict(pool[lab["origin_id"]]);r["label"]=lab;rows.append(r)
 pubs=json.loads((BASE/"research/expansion_v2_sources/verification.json").read_text())
 for lab in json.loads((BASE/"research/expansion_v2_sources/annotations.json").read_text()):
  r=dict(next(x for x in pubs if x["source_id"]==lab["origin_id"]))
  r.update(origin_id=lab["origin_id"],family="source-"+lab["family"],label=lab);rows.append(r)
 hr=BASE/"research/expansion_v2_hard_retry"
 info=json.loads((hr/"result.json").read_text())
 if info["status"]=="verified":
  inp=next(x for x in json.loads((BASE/"research/pilot100/public-inputs.json").read_text()) if x[0]=="33iq-507273")
  lab=json.loads((hr/"annotation.json").read_text())
  rows.append(dict(origin_id=inp[0],family="33iq-507273",source="public_position",hands=inp[1],turn=inp[2],last_text=inp[3],passes=inp[4],source_url=inp[5],source_note="Previously unresolved source; fresh three-way proof completed",label=lab))
 for row in rows:
  label=row["label"]
  if "target" in label:row["label"]={**label,**label["target"]}
 blocked=read_blocked()
 rng=random.Random(2026110508)
 controls=[]
 # 15 exact per-seat card-count pairs, prioritizing the reviewed grade-4 examples.
 parents=sorted([r for r in rows if r["label"]["overall"]>=3 and r.get("last") is None and not r.get("last_text") and hand_size(restore(r).hands[r["turn"]])<=12],key=lambda r:(-r["label"]["overall"],r["origin_id"]))[:15]
 pairs=[]
 for parent in parents:
  p=restore(parent);counts=[hand_size(h) for h in p.hands];n=counts[p.turn]
  for attempt in range(200):
   active=list(range(3,3+n)) if n>=5 else [10]*n
   deck=full_deck()
   for card in active:deck.remove(card)
   rng.shuffle(deck);hs=[];k=0
   for seat,num in enumerate(counts):
    if seat==p.turn:hs.append(hand_from_cards(active))
    else:hs.append(hand_from_cards(deck[k:k+num]));k+=num
   cp=Position(tuple(hs),p.turn)
   if state_key(cp) in blocked:continue
   cr=dict(origin_id=parent["origin_id"]+"-count-control",family=parent["family"],source="synthetic_exact_count_control",hands=[hand_to_text(h) for h in hs],turn=p.turn,last=None,passes=0,
           parent_origin=parent["origin_id"],construction="same three seat counts and turn; independently redealt cards; active hand is a one-move finish; NOT same multiset")
   cr["label"]=dict(overall=1,reasoning_depth=1,move_concealment=1,reason="行动方整手可一次合法出完；直接结束，无须推演其他人的应对。",mechanism="一手出完对照")
   assert any(cp.winner_after(m) for m in cp.legal())
   blocked.add(state_key(cp));controls.append(cr)
   pairs.append({"hard_origin":parent["origin_id"],"easy_origin":cr["origin_id"],"counts":counts,"turn":p.turn,"kind":"engineered_exact_per_seat_counts","same_card_multiset":False,"split_group":parent["family"]})
   break
  else:raise RuntimeError("could not make unique control")
 rows+=controls
 seen=read_blocked();dataset=[];evidence=[]
 for i,r in enumerate(rows,201):
  p=validate_three(r);key=state_key(p)
  assert key not in seen,r["origin_id"];seen.add(key)
  lab=r["label"]
  ident=f"EX{i:03d}"
  inp={"hands":[hand_to_cards(h) for h in p.hands],"turn":p.turn,"last":dict(cards=list(p.last.cards),type=int(p.last.type),main=p.last.main,length=p.last.length) if p.last else None,"passes":p.passes,"rank_encoding":"3..14=3..A;15=2;16=small_joker;17=big_joker","roles":["landlord","farmer_down","farmer_up"]}
  dataset.append(dict(id=ident,input=inp,target={k:lab[k] for k in ("overall","reasoning_depth","move_concealment")},reason=lab["reason"],mechanism=lab.get("mechanism"),annotation={"type":"assistant_provisional_estimate","rubric":"v0.1","human_calibrated":False,"blind":False,"date":"2026-11-05","evidence_notes":lab.get("evidence_notes")} ,
    provenance={"source":r["source"],"origin_id":r["origin_id"],"family":r["family"],"url":r.get("source_url"),"note":r.get("source_note"),"parent_origin":r.get("parent_origin"),"construction":r.get("construction")},
    verification={"status":"verified","key":key,"winning_moves":r["winning_moves"],"legal_move_count":len(r["moves"]),"evidence":"evidence.json","exact_under":"vendored bdee2f6e; optimized, canonical-off and independent enumerator root outcomes agree"}))
  evidence.append(dict(id=ident,**r));print(ident,r["origin_id"],lab["overall"],"verified",round(r["fresh_verification"]["seconds"],2),flush=True)
  dump(OUT/"verification-progress.json",evidence)
 byorigin={r["provenance"]["origin_id"]:r for r in dataset}
 for i,pair in enumerate(pairs,1):
  pair.update(id=f"PAIR{i:02d}",hard_id=byorigin[pair["hard_origin"]]["id"],easy_id=byorigin[pair["easy_origin"]]["id"],grade_gap=byorigin[pair["hard_origin"]]["target"]["overall"]-1)
 (OUT/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dataset),encoding="utf-8")
 dump(OUT/"evidence.json",evidence);dump(OUT/"pairs.json",pairs)
 dump(OUT/"manifest.json",dict(rows=len(dataset),grades=dict(Counter(r["target"]["overall"] for r in dataset)),pairs=len(pairs),three_way_verified=len(evidence),human_calibrated=False,training_changed=False,dataset_sha256=hashlib.sha256((OUT/"dataset.jsonl").read_bytes()).hexdigest(),python=sys.version,platform=platform.platform(),seeds=[2026110507,2026110508],excluded_from_old_splits=True))
 (OUT/"verification-progress.json").unlink()
if __name__=="__main__":main()
