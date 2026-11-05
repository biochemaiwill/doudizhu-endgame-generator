import sys,json,hashlib
from pathlib import Path
from collections import Counter
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
OUT=BASE/"research/expansion_v3_sources"
rows=json.loads((OUT/"candidates.json").read_text())
rows += [r for r in json.loads((OUT/"swap-candidates.json").read_text()) if r["origin_id"]=="PUB3-0907-swap-11"]
labs=json.loads((OUT/"labels.json").read_text())
src={r["source_id"]:r for r in json.loads((OUT/"source-inputs.json").read_text())}
blocked=set()
for path in [BASE/"research/pilot100/dataset.jsonl",BASE/"research/expansion_v2/dataset.jsonl",BASE/"research/highscore_candidates_v1/candidates.jsonl"]:
 for r in map(json.loads,path.read_text().splitlines()):blocked.add(r.get("verification",{}).get("key",r.get("key")))
for r in json.loads((BASE/"research/grade5_search_v3/verified-candidates.json").read_text()):blocked.add(r["key"])
byid={r["origin_id"]:r for r in rows};audit=[]
for r in rows:
 r["label"]=labs[r["origin_id"]];r["label"].update(rubric="v0.1",human_calibrated=False,blind=False,type="assistant_provisional_estimate")
 r["reason"]=r["label"]["reason"]
 r["mechanism"]=r["label"]["mechanism"]
 q=restore(r);validate_position(q)
 if "history" in r:
  p=restore(src[r["family"]])
  for h in r["history"]:
   assert p.turn==h["player"]
   mv=[m for m in p.legal() if int(m.type)==h["type"] and list(m.cards)==h["cards"]]
   assert len(mv)==1 and not p.winner_after(mv[0]);p=p.play(mv[0])
  assert p==q
 else:
  parent=restore(byid[r["parent"]]);ph=[hand_to_cards(h) for h in parent.hands];qh=[hand_to_cards(h) for h in q.hands]
  assert list(map(len,ph))==list(map(len,qh));assert Counter(sum(ph,[]))==Counter(sum(qh,[]))
  assert parent.turn==q.turn and parent.last==q.last and parent.passes==q.passes
 sig=r["root_signatures"];assert sig["optimized"]==sig["canonical_off"]==sig["independent"]
 assert {(int(m.type),tuple(m.cards)) for m in q.legal()}=={(s[0],tuple(s[1])) for s in sig["optimized"]}
 assert r["key"] not in blocked;blocked.add(r["key"]);assert not any(q.winner_after(m) for m in q.legal())
 audit.append(dict(origin_id=r["origin_id"],status="pass",three_way=True,prefix_or_swap_legal=True,prior_overlap=False,no_immediate_finish=True))
pair=dict(id="SRC3-PAIR01",hard_origin="PUB3-0907",easy_origin="PUB3-0907-swap-11",hard_overall=3,easy_overall=2,per_seat_counts=[7,12,4],same_card_multiset=True,same_turn=True,same_last=True,same_passes=True,family=byid["PUB3-0907"]["family"])
for name,value in [("selection.json",rows),("pairs.json",[pair]),("validation.json",audit)]:
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
m=json.loads((OUT/"manifest.json").read_text());m.update(selected=4,selected_grades={"2":3,"3":1},pairs=1,grade5_confirmed=0,training_changed=False,sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ["selection.json","pairs.json","validation.json"]})
(OUT/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2))
print("PASS 4 rows; legal prefixes/swap; one exact-count and exact-multiset pair; no immediate finish")
