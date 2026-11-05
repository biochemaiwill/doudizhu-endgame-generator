from pathlib import Path
import json,sys,collections
B=Path("/home/wutianxiang/doudizhu-endgame-generator");sys.path.insert(0,str(B/"research/pilot100"))
from build import *
rows={r["id"]:r for r in map(json.loads,(B/"research/mechanism_pairs_v1/dataset.jsonl").read_text().splitlines())}
out={}
for rid in ["NX325","NX323"]:
 r=rows[rid];start=Position(tuple(hand_from_cards(h) for h in r["input"]["hands"]),r["input"]["turn"])
 s=Solver(node_limit=3000000);side=start.side(start.turn);queue=collections.deque([(start,[])]);seen=set();found=None
 while queue and len(seen)<5000:
  p,path=queue.popleft();key=(p.hands,p.turn,p.last,p.passes)
  if key in seen:continue
  seen.add(key);vals=s.move_values(p)
  if p.side(p.turn)==side:
   bad=[m for m,w in vals if not w]
   if bad:found={"prefix":path,"wrong":bad[0].text(),"all_moves":[move_record(m,w) for m,w in vals]};break
  if len(path)>=14:continue
  for m,w in vals:
   if p.side(p.turn)==side and not w:continue
   if p.winner_after(m):continue
   queue.append((p.play(m),path+[m.text()]))
 out[rid]={"visited":len(seen),"found":found};print(rid,out[rid],flush=True)
(B/"research/mechanism_pairs_v1/extra_mistake_search.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
