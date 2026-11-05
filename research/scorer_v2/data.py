"""Merge frozen datasets and split source/pair connected components."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import urlsplit,urlunsplit
import numpy as np
ROOT=Path(__file__).resolve().parent
INPUTS=[(ROOT.parent/"pilot100/dataset.jsonl","85b3d83ca0ed203801161299c83178f858b8e0d3e3a6a288341ca759ada0b59c"),(ROOT.parent/"expansion_v2/dataset.jsonl","fc4f47a3e922aca4a8509d30f8524f82d4f90bbdfa0fd7375f5f2516cfcab5fb")]
SEED=2026110511
def load_rows():
 rows=[]
 for path,digest in INPUTS:
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
  rows.extend(json.loads(x) for x in path.read_text().splitlines())
 assert len(rows)==153 and len({r["id"] for r in rows})==153
 assert len({r["verification"]["key"] for r in rows})==153
 return rows
def normalize_url(url):
 if not url:return None
 p=urlsplit(url)
 return urlunsplit((p.scheme.lower(),p.netloc.lower().removeprefix("www."),p.path.rstrip("/"),p.query,""))
def components(rows,pairs):
 parent={r["id"]:r["id"] for r in rows}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 buckets=defaultdict(list)
 for r in rows:
  family=r["provenance"].get("family")
  if family:buckets[("family",family)].append(r["id"])
  url=normalize_url(r["provenance"].get("url"))
  if url:buckets[("source_url",url)].append(r["id"])
 for ids in buckets.values():
  for i in ids[1:]:union(ids[0],i)
 for p in pairs:union(p["hard_id"],p["easy_id"])
 groups=defaultdict(list)
 for r in rows:groups[find(r["id"])].append(r["id"])
 return {k:sorted(v) for k,v in groups.items()}
def subset(groups,counts,target,rng):
 keys=list(groups);rng.shuffle(keys);zero=(0,0,0,0);dp={zero:()}
 target=tuple(target)
 for k in keys:
  v=counts[k]
  for state,path in list(dp.items()):
   nxt=tuple(a+b for a,b in zip(state,v))
   if all(a<=b for a,b in zip(nxt,target)) and nxt not in dp:dp[nxt]=path+(k,)
  if target in dp:return set(dp[target])
 return None
def make_split(rows,pairs):
 byid={r["id"]:r for r in rows};groups=components(rows,pairs)
 counts={g:tuple(sum(byid[i]["target"]["overall"]==k for i in ids) for k in range(1,5)) for g,ids in groups.items()}
 anchored={g for g,ids in groups.items() if any(i.startswith("DZ") for i in ids)}
 eligible={g:ids for g,ids in groups.items() if g not in anchored}
 rng=np.random.default_rng(SEED);found=False
 for attempt in range(500):
  va=subset(eligible,counts,(3,2,6,1),rng)
  if va is None:continue
  te=subset({g:ids for g,ids in eligible.items() if g not in va},counts,(4,3,8,2),rng)
  if te is not None:found=True;break
 if not found:raise ValueError("No exact grouped stratification meets prespecified targets")
 assign={g:("val" if g in va else "test" if g in te else "train") for g in groups}
 parts={p:sorted(i for g,ids in groups.items() if assign[g]==p for i in ids) for p in ("train","val","test")}
 # Assertions are part of the split contract, not post-hoc selection.
 assert all(i.startswith("EX") for i in parts["val"]+parts["test"])
 assert all(sum(i in parts[p] for p in parts)==1 for i in byid)
 return {"seed":SEED,"ids":parts,"group_members":groups,"group_assignment":assign,"grouping":["provenance.family","normalized identical source URL","explicit pair connected components"],"old_connected_groups_anchored_train":sorted(anchored),"counts":{p:{str(k):sum(byid[i]["target"]["overall"]==k for i in ids) for k in range(1,5)} for p,ids in parts.items()},"test_status":"new-batch source/pair grouped development holdout, not human-calibrated external final test"}
def class_weights(y):
 counts=Counter(map(int,y))
 if set(counts)!={1,2,3,4}:raise ValueError("all four grades required")
 return np.array([len(y)/(4*counts[int(v)]) for v in y],dtype=np.float32)
def support_ids(rows,train_ids,seed,k=2):
 byid={r["id"]:r for r in rows};rng=np.random.default_rng(seed);out=[]
 for g in range(1,5):
  choices=sorted(i for i in train_ids if byid[i]["target"]["overall"]==g)
  if len(choices)<k:raise ValueError("insufficient support")
  out+=rng.choice(choices,k,replace=False).tolist()
 return sorted(out)
def main():
 rows=load_rows();pairs=json.loads((ROOT.parent/"expansion_v2/pairs.json").read_text())
 split=make_split(rows,pairs)
 if (ROOT/"split.json").exists():raise SystemExit("Refuse overwrite frozen split")
 (ROOT/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
 (ROOT/"split.json").write_text(json.dumps(split,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"n":len(rows),"counts":split["counts"],"groups":len(split["group_members"]),"ids":split["ids"]},ensure_ascii=False))
if __name__=="__main__":main()
