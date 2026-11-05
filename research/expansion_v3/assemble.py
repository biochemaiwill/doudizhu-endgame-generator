from pathlib import Path
import sys,json,signal,hashlib
from collections import Counter
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator");OUT=BASE/"research/expansion_v3"
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def jsonl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def keys(x):
 if isinstance(x,dict):return ({x["key"]} if isinstance(x.get("key"),str) and len(x["key"])==64 else set())|set().union(*(keys(v) for v in x.values()))
 if isinstance(x,list):return set().union(*(keys(v) for v in x))
 return set()
def main():
 if (OUT/"dataset.jsonl").exists():raise SystemExit("refuse overwrite")
 old=jsonl(BASE/"research/scorer_v2/dataset.jsonl");oldby={r["id"]:r for r in old}
 blocked={r["verification"]["key"] for r in old}
 blocked|=keys(jsonl(BASE/"research/highscore_candidates_v1/candidates.jsonl"))
 blocked|=keys(json.loads((BASE/"research/grade5_search_v3/verified-candidates.json").read_text()))
 pool=jsonl(BASE/"research/expansion_v3_synthetic/formal_candidates.jsonl")
 candidates=json.loads((BASE/"research/expansion_v3_pairs/candidates.json").read_text())
 for lab in json.loads((BASE/"research/expansion_v3_pairs/labels.json").read_text()):
  r=dict(next(r for r in candidates if r["origin_id"]==lab["origin_id"]));r.update(label=lab["label"],reason=lab["reason"],mechanism=lab["mechanism"],certainty_notes=lab["certainty_notes"]);pool.append(r)
 source=BASE/"research/expansion_v3_sources/selection.json"
 if source.exists():pool+=json.loads(source.read_text())
 pool.sort(key=lambda r:(-r["label"]["overall"],r["origin_id"]))
 result=[];evidence=[];pairs=[]
 split=json.loads((BASE/"research/scorer_v2/split.json").read_text())["ids"]
 for i,r in enumerate(pool,301):
  p=restore(r);validate_position(p);key=state_key(p)
  assert key not in blocked,r["origin_id"];blocked.add(key)
  signal.setitimer(signal.ITIMER_REAL,45)
  try:
   vals=Solver(node_limit=5000000).move_values(p);sig=signature(vals)
   assert sig==signature(Solver(node_limit=5000000,canonical=False).move_values(p))
   assert [((t,c),w) for t,c,w in sig]==ref.move_values(p)
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  wins=[m.text() for m,w in vals if w];assert wins
  lab=r["label"];ident=f"NX{i:03d}"
  inp=dict(hands=[hand_to_cards(h) for h in p.hands],turn=p.turn,last=dict(cards=list(p.last.cards),type=int(p.last.type),main=p.last.main,length=p.last.length) if p.last else None,passes=p.passes,rank_encoding="3..14=3..A;15=2;16=small_joker;17=big_joker",roles=["landlord","farmer_down","farmer_up"])
  parent=r.get("parent_id")
  row=dict(id=ident,input=inp,target={k:lab[k] for k in ("overall","reasoning_depth","move_concealment")},reason=r["reason"],mechanism=r.get("mechanism"),
   annotation=dict(type="assistant_provisional_estimate",rubric="v0.1",human_calibrated=False,blind=False,date="2026-11-05",review_notes=r.get("certainty_notes")),
   provenance=dict(source=r.get("source","public_source_derived"),origin_id=r["origin_id"],family=r["family"],url=r.get("source_url"),parent_id=parent,note=r.get("source_note")),
   verification=dict(status="verified",key=key,winning_moves=wins,legal_move_count=len(vals),evidence="evidence.json"))
  if parent:
   a=oldby[parent]["input"]
   assert [len(h) for h in a["hands"]]==[len(h) for h in inp["hands"]]
   assert sorted(sum(a["hands"],[]))==sorted(sum(inp["hands"],[]))
   assert all(a[k]==inp[k] for k in ["turn","last","passes"])
   assert not any(p.winner_after(m) for m in p.legal())
   pairs.append(dict(id=f"CF{len(pairs)+1:03d}",hard_id=parent,easy_id=ident,same_per_seat_counts=True,same_rank_multiset=True,no_immediate_finish=True,grade_gap=oldby[parent]["target"]["overall"]-lab["overall"],family=r["family"],prior_parent_split=next(s for s,ids in split.items() if parent in ids)))
  result.append(row);evidence.append(dict(id=ident,source_record=r,verified_moves=[move_record(m,w) for m,w in vals]));print(ident,r["origin_id"],lab["overall"],"PASS",flush=True)
 byorigin={r["provenance"]["origin_id"]:r for r in result};sp=BASE/"research/expansion_v3_sources/pairs.json"
 if sp.exists():
  for item in json.loads(sp.read_text()):
   hard=byorigin[item["hard_origin"]];easy=byorigin[item["easy_origin"]]
   a=hard["input"];b=easy["input"]
   assert [len(h) for h in a["hands"]]==[len(h) for h in b["hands"]]
   assert sorted(sum(a["hands"],[]))==sorted(sum(b["hands"],[]))
   assert all(a[k]==b[k] for k in ["turn","last","passes"])
   pairs.append(dict(id=f"CF{len(pairs)+1:03d}",hard_id=hard["id"],easy_id=easy["id"],same_per_seat_counts=True,same_rank_multiset=True,no_immediate_finish=True,grade_gap=hard["target"]["overall"]-easy["target"]["overall"],family=hard["provenance"]["family"],prior_parent_split=None))
 (OUT/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in result))
 dump(OUT/"evidence.json",evidence);dump(OUT/"pairs.json",pairs)
 dump(OUT/"manifest.json",dict(rows=len(result),grades=dict(Counter(r["target"]["overall"] for r in result)),paired_groups=len(pairs),sha256=hashlib.sha256((OUT/"dataset.jsonl").read_bytes()).hexdigest(),old_data_unchanged_sha256=hashlib.sha256((BASE/"research/scorer_v2/dataset.jsonl").read_bytes()).hexdigest(),entered_old_training=False))
if __name__=="__main__":main()
