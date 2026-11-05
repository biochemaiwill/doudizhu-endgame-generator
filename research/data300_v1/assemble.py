"""Assemble reviewed few-hundred corpus without overwriting any prior dataset."""
from pathlib import Path
import sys,json,hashlib,signal,gzip
from collections import Counter
B=Path("/home/wutianxiang/doudizhu-endgame-generator/research");R=B/"data300_v1"
sys.path.insert(0,str(B/"pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
def jl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def main():
 if (R/"dataset_merged.jsonl").exists():raise SystemExit("refuse overwrite")
 old=jl(B/"scorer_v4/dataset.jsonl")+jl(B/"grade4_expansion_v5/dataset.jsonl")
 oldkeys={r["verification"]["key"] for r in old}
 for r in jl(B/"mechanism_pairs_v1/dataset.jsonl"):
  if r["verification"]["key"] not in oldkeys:old.append(r);oldkeys.add(r["verification"]["key"])
 assert len(old)==190
 pool=json.loads((R/"candidate_pool.json").read_text() if (R/"candidate_pool.json").exists() else gzip.open(R/"candidate_pool.json.gz","rt",encoding="utf-8").read());byorigin={r["origin_id"]:r for r in pool}
 labels=[]
 for i in range(3):labels+=json.loads((R/("labels0_zh.json" if i==0 else f"labels{i}.json")).read_text())
 assert len(labels)==len(pool) and len({r["origin_id"] for r in labels})==len(pool)
 lm={r["origin_id"]:r for r in labels};special=jl(B/"mechanism_pairs_v2/dataset.jsonl")
 conflicts=[];conflict_keys=set();rescued=[]
 for row in special:
  key=row["verification"]["key"];orig=row["provenance"]["origin_id"]
  if row["provenance"]["source"]=="new_swap_counterfactual":continue
  lab=lm.get(orig)
  if lab and lab["overall"] is not None and lab["overall"]!=row["target"]["overall"]:
   conflict_keys.add(key);conflicts.append({"origin_id":orig,"key":key,"review_a":lab,"review_b":row,"decision":"quarantine_no_hard_label"})
  elif lab and lab["overall"] is None:rescued.append({"origin_id":orig,"decision":"accepted_after_additional_mechanism_review","evidence":"../mechanism_pairs_v2/mechanism_evidence.json"})
 accepted=[];rejected=[]
 for lab in labels:
  row=byorigin[lab["origin_id"]]
  if row["key"] in conflict_keys:continue
  if lab["overall"] is None:rejected.append(lab);continue
  accepted.append((row,lab))
 new=[];evidence=[];seen=set(oldkeys);serial=1
 for row,lab in accepted:
  p=restore(row);validate_position(p)
  signal.setitimer(signal.ITIMER_REAL,20)
  try:
   vals=Solver(node_limit=2000000).move_values(p);sig=signature(vals)
   assert sig==signature(Solver(node_limit=2000000,canonical=False).move_values(p))
   assert [((t,c),w) for t,c,w in sig]==ref.move_values(p)
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  key=state_key(p);assert key not in seen;seen.add(key);assert any(w for m,w in vals)
  ident=f"D300-{serial:03d}";serial+=1
  inp={"hands":[hand_to_cards(h) for h in p.hands],"turn":p.turn,"last":dict(cards=list(p.last.cards),type=int(p.last.type),main=p.last.main,length=p.last.length) if p.last else None,"passes":p.passes,"rank_encoding":"3..14=3..A;15=2;16=small_joker;17=big_joker","roles":["landlord","farmer_down","farmer_up"]}
  new.append(dict(id=ident,input=inp,target={k:lab[k] for k in ["overall","reasoning_depth","move_concealment"]},reason=lab["reason"],mechanism=lab["mechanism"],annotation=dict(type="assistant_weak_label",rubric="v0.1",human_calibrated=False,blind_to_model_predictions=True,review_status=lab["review_status"],evidence_note=lab["evidence_note"]),provenance=dict(source="reviewed_independent_synthetic",family=row["family"],origin_id=row["origin_id"],pool_source=row["pool_source"]),verification=dict(status="verified",key=key,winning_moves=[m.text() for m,w in vals if w],legal_move_count=len(vals),evidence="new_evidence.json")))
  evidence.append(dict(id=ident,source_record=row,label_review=lab,root_signature=sig))
  if serial%25==0:print("verified",serial-1,flush=True)
 for row in special:
  key=row["verification"]["key"]
  if key in conflict_keys or key in seen:continue
  seen.add(key);new.append(row)
 merged=old+new;assert len({r["id"] for r in merged})==len(merged)
 pairs=[];pairseen=set()
 for f in [B/"expansion_v2/pairs.json",B/"expansion_v3/pairs.json",B/"mechanism_pairs_v1/pairs.json",B/"mechanism_pairs_v2/pairs.json"]:
  for p in json.loads(f.read_text()):
   edge=tuple(sorted([p["hard_id"],p["easy_id"]]))
   ids={r["id"] for r in merged}
   if not set(edge)<=ids or edge in pairseen:continue
   pairseen.add(edge);pairs.append(p)
 for name,rows in [("dataset_new.jsonl",new),("dataset_merged.jsonl",merged)]:(R/name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
 save(R/"new_evidence.json",evidence);save(R/"quarantined_conflicts.json",conflicts);save(R/"rejected.json",rejected);save(R/"rescued_reviews.json",rescued);save(R/"pairs.json",pairs)
 save(R/"manifest.json",dict(old_rows=len(old),new_rows=len(new),merged_rows=len(merged),distribution=dict(Counter(r["target"]["overall"] for r in merged)),conflicts_quarantined=len(conflicts),rejected_reviews=len(rejected),rescued_reviews=len(rescued),pairs=len(pairs),human_calibrated=False,newly_reviewed_three_way=len(evidence),sha256=hashlib.sha256((R/"dataset_merged.jsonl").read_bytes()).hexdigest()))
 print((R/"manifest.json").read_text(),flush=True)
if __name__=="__main__":main()
