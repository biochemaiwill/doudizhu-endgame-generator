"""New source-family candidates from deterministic legal prefixes. No training."""
import sys,json,random,time,signal,platform,gzip,hashlib
from pathlib import Path
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/"research/expansion_v3_sources"
src=[r for r in json.loads((BASE/"research/grade5_search_v3/source-inputs.json").read_text()) if r["source_id"]!="baidu-20131104-bounty"]
blocked=set()
for path in [BASE/"research/pilot100/dataset.jsonl",BASE/"research/expansion_v2/dataset.jsonl",BASE/"research/highscore_candidates_v1/candidates.jsonl"]:
 for r in map(json.loads,path.read_text().splitlines()):blocked.add(r.get("verification",{}).get("key",r.get("key")))
for r in json.loads((BASE/"research/grade5_search_v3/verified-candidates.json").read_text()):blocked.add(r["key"])
def run():
 if (OUT/"candidates.json").exists():raise SystemExit("refuse overwrite")
 rng=random.Random(2026110521);t0=time.monotonic();out=[];ledger=[];counts=Counter()
 for i in range(1000):
  if time.monotonic()-t0>170 or len(out)>=14:break
  source=src[i%len(src)];p=restore(source);history=[]
  assert [hand_size(h) for h in p.hands]==[20,17,17]
  target=rng.randint(19,26)
  for _ in range(140):
   if sum(map(hand_size,p.hands))<=target:break
   choices=[m for m in p.legal() if not p.winner_after(m)]
   if not choices:break
   weights=[1+max(0,m.n-1)*(1 if i%3==0 else 3) for m in choices]
   m=rng.choices(choices,weights=weights)[0];history.append(dict(player=p.turn,**move_record(m)));p=p.play(m)
  r=dict(origin_id=f'PUB3-{i:04d}',family=source["source_id"],source="synthetic_legal_prefix_from_complete_public_deal",source_url=source["source_url"],history=history,hands=[hand_to_text(h) for h in p.hands],turn=p.turn,last=move_record(p.last) if p.last else None,passes=p.passes,key=state_key(p))
  if r["key"] in blocked:continue
  blocked.add(r["key"])
  try:
   validate_position(p);signal.setitimer(signal.ITIMER_REAL,1.3)
   s=Solver(node_limit=180000);vals=s.move_values(p);wins=[m for m,w in vals if w]
   if not wins:r["status"]="losing"
   elif len(vals)<4 or len(wins)>2 or any(p.winner_after(m) for m in wins):r["status"]="not_narrow"
   else:
    signal.setitimer(signal.ITIMER_REAL,6)
    a=signature(vals);b=signature(Solver(node_limit=700000,canonical=False).move_values(p));c=[(t,c,w) for ((t,c),w) in ref.move_values(p)]
    assert a==b==c
    r.update(status="verified",moves=[move_record(m,w) for m,w in vals],winning_moves=[m.text() for m in wins],root_signatures={"optimized":a,"canonical_off":b,"independent":c},pv=trace(p,s,maxplies=80))
    r["trap_examples"]=[dict(first=m.text(),continuation=trace(p.play(m),s,maxplies=80)) for m,w in vals if not w][:3]
    out.append(r)
   rec=dict(r) if r.get("status")=="verified" else {k:r[k] for k in ["origin_id","family","key","status","hands","turn","last","passes"]}
  except (TimeBudget,BudgetExceeded) as e:rec=dict(origin_id=r["origin_id"],family=r["family"],key=r["key"],status="unknown_budget",error=type(e).__name__)
  except Exception as e:rec=dict(origin_id=r["origin_id"],family=r["family"],key=r["key"],status="error",error=repr(e))
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  counts[rec["status"]]+=1;ledger.append(rec)
  if i%50==0:print(i,len(out),round(time.monotonic()-t0),dict(counts),flush=True)
 (OUT/"candidates.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
 (OUT/"source-inputs.json").write_text(json.dumps(src,ensure_ascii=False,indent=2))
 raw="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in ledger).encode()
 (OUT/"search-ledger.jsonl.gz").write_bytes(gzip.compress(raw,mtime=0))
 (OUT/"manifest.json").write_text(json.dumps(dict(seed=2026110521,attempted=len(ledger),candidates=len(out),elapsed=time.monotonic()-t0,counts=dict(counts),rules="unchanged bdee2f6e",python=sys.version,platform=platform.platform(),human_calibrated=False,source_original_win_status="unknown; generated prefixes individually certified",uncompressed_ledger_sha256=hashlib.sha256(raw).hexdigest()),indent=2))
 print("DONE",len(out),dict(counts),flush=True)
if __name__=="__main__":run()
