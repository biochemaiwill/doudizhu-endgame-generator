"""Rebuild the frozen mixed-source selection from retained candidate evidence."""
import json
from pathlib import Path
from collections import defaultdict,deque
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/"selection.synthetic-initial.json").read_text())
hard=json.loads((ROOT/"hard-pool.json").read_text())
hi=[0,1,5,7,9,10,13,14,17,18,19,26]
chosen=[]
for r in base[:30]:
 r["origin_id"]=r["id"];r["source"]="synthetic_baseline";chosen.append(r)
chosen += [dict(hard[i],source="synthetic_enriched",origin_id="H"+str(i+1)) for i in hi]
public=[dict(r,origin_id=r["source_id"]) for r in json.loads((ROOT/"public-ordered-verification.json").read_text()) if r["status"]=="verified"]
seen={r["key"] for r in chosen+public};buckets=defaultdict(deque)
for r in json.loads((ROOT/"source-verification.json").read_text()):
 if r["status"]!="verified" or r["source"]=="public_position" or r["key"] in seen:continue
 seen.add(r["key"])
 k=(r["last"]["type"] if r["last"] else 0,len(r["winning_moves"])==1,r["n_cards"]//5)
 buckets[k].append(r)
master=[]
while len(master)<50 and any(buckets.values()):
 for k in sorted(buckets):
  if buckets[k] and len(master)<50:master.append(buckets[k].popleft())
for i,r in enumerate(master,1):r["origin_id"]="M"+str(i)
chosen+=master+public
assert len(chosen)==100 and len({r["key"] for r in chosen})==100
for i,r in enumerate(chosen,1):r["id"]=f"DZ{i:03d}"
(ROOT/"selection.json").write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("100 frozen mixed-source positions rebuilt")
