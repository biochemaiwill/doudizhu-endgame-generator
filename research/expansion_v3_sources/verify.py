"""Read-only proof/serialization checks for source-derived candidates."""
from pathlib import Path
import sys,json,hashlib
BASE=Path("/home/wutianxiang/doudizhu-endgame-generator")
sys.path.insert(0,str(BASE/"research/pilot100"))
from build import *
from verify_sources import restore
OUT=BASE/"research/expansion_v3_sources"
def main():
 rows=json.loads((OUT/"candidates.json").read_text());src={r["source_id"]:r for r in json.loads((OUT/"source-inputs.json").read_text())}
 seen=set();audit=[]
 for r in rows:
  p=restore(src[r["family"]]);validate_position(p)
  assert [hand_size(h) for h in p.hands]==[20,17,17]
  for step in r["history"]:
   assert p.turn==step["player"]
   moves=[m for m in p.legal() if int(m.type)==step["type"] and list(m.cards)==step["cards"]]
   assert len(moves)==1 and not p.winner_after(moves[0])
   p=p.play(moves[0])
  assert p==restore(r);key=state_key(p);assert key==r["key"] and key not in seen;seen.add(key)
  sigs=r["root_signatures"];assert sigs["optimized"]==sigs["canonical_off"]==sigs["independent"]
  expected={(int(m.type),tuple(m.cards)) for m in p.legal()}
  assert {(s[0],tuple(s[1])) for s in sigs["optimized"]}==expected
  assert any(s[2] for s in sigs["optimized"])
  audit.append(dict(origin_id=r["origin_id"],prefix_legal=True,root_signatures_equal=True,unique=True))
 (OUT/"validation.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 print("PASS",len(rows))
if __name__=="__main__":main()
