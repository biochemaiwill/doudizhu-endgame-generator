"""Recheck source-derived candidates and every recorded legal prefix."""
from pathlib import Path
import sys,json,signal,hashlib
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/"research/grade5_search_v3"
sources={r["source_id"]:r for r in json.loads((OUT/"source-inputs.json").read_text())}
prior=[]
for path in [BASE/"research/pilot100/dataset.jsonl",BASE/"research/expansion_v2/dataset.jsonl",BASE/"research/highscore_candidates_v1/candidates.jsonl"]:
 prior+=list(map(json.loads,path.read_text().splitlines()))
blocked={r.get("verification",{}).get("key",r.get("key")) for r in prior}
audit=[]
for r in json.loads((OUT/"verified-candidates.json").read_text()):
 p=restore(sources[r["family"]])
 for h in r["history"]:
  assert p.turn==h["player"]
  matches=[m for m in p.legal() if int(m.type)==h["type"] and list(m.cards)==h["cards"]]
  assert len(matches)==1
  assert not p.winner_after(matches[0])
  p=p.play(matches[0])
 q=restore(r);assert p==q;validate_position(q)
 assert state_key(q) not in blocked
 signal.setitimer(signal.ITIMER_REAL,20)
 try:
  a=signature(Solver(node_limit=1000000).move_values(q))
  b=signature(Solver(node_limit=1000000,canonical=False).move_values(q))
  c=[(t,c,w) for ((t,c),w) in ref.move_values(q)]
  assert a==b==c
  audit.append({"source_id":r["source_id"],"key":state_key(q),"prefix_legal":True,"prior_overlap":False,"optimized_signature":a,"canonical_off_signature":b,"independent_signature":c,"status":"pass"})
 finally:signal.setitimer(signal.ITIMER_REAL,0)
(OUT/"validation.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print("PASS",len(audit),"legal prefixes and three-way root signatures")
