from pathlib import Path
import sys,json,gzip,random,hashlib
import numpy as np
B=Path("/home/wutianxiang/doudizhu-endgame-generator");R=B/"research/scorer_v5"
sys.path.insert(0,str(B/"research/pilot100"))
from build import Position,hand_from_cards,hand_to_cards,state_key,full_deck,Solver,signature
import reference_ordered as ref
with gzip.open(R/"pretrain.json.gz","rt") as f:data=json.load(f)
blocked={json.loads(s)["verification"]["key"] for s in (B/"research/scorer_v4/dataset.jsonl").read_text().splitlines()}
ks=[r["key"] for r in data];assert len(ks)==len(set(ks)) and not blocked.intersection(ks)
for row in data:
 x=np.array(row["input"]);hands=[sum(([i+3]*int(round(c*4)) for i,c in enumerate(h)),[]) for h in x[:45].reshape(3,15)]
 p=Position(tuple(hand_from_cards(h) for h in hands),int(x[45:48].argmax()))
 assert state_key(p)==row["key"]
 legal={(int(m.type),tuple(m.cards)) for m in p.legal()}
 actions={(int(np.array(a)[:15].argmax()),tuple(sum(([i+3]*int(round(c*4)) for i,c in enumerate(a[17:])),[]))) for a in row["actions"]}
 assert legal==actions
 assert len(row["labels"])==len(legal)
# Independently recheck every root label on a fixed sample of 20 states.
for i in np.random.default_rng(20261105).choice(len(data),20,replace=False):
 row=data[i];x=np.array(row["input"]);hands=[sum(([j+3]*int(round(c*4)) for j,c in enumerate(h)),[]) for h in x[:45].reshape(3,15)]
 p=Position(tuple(hand_from_cards(h) for h in hands),int(x[45:48].argmax()))
 vals=ref.move_values(p);given=sorted((int(np.array(a)[:15].argmax()),tuple(sum(([j+3]*int(round(c*4)) for j,c in enumerate(a[17:])),[])),bool(y)) for a,y in zip(row["actions"],row["labels"]))
 assert given==[(t,c,w) for (t,c),w in vals]
# Recreate only RNG draws, without rerunning time-limited searches.
rng=random.Random(2026110518);known=set(blocked);cursor=0;attempts=0
while cursor<len(ks):
 deck=full_deck();rng.shuffle(deck);sizes=[rng.randint(3,6) for _ in range(3)];hs=[];k=0
 for size in sizes:hs.append(hand_from_cards(deck[k:k+size]));k+=size
 key=state_key(Position(tuple(hs),rng.randrange(3)));attempts+=1
 if key in known:continue
 known.add(key)
 if key==ks[cursor]:cursor+=1
 assert attempts<200000
assert len(data)==1000,"cannot exactly recover end RNG if original stopped by wall time"
order=list(range(len(data)));rng.shuffle(order);valid=set(order[:len(data)//5])
def summary(ids):
 labs=[y for i in ids for y in data[i]["labels"]];n=len(labs);pos=sum(labs)
 return dict(states=len(ids),actions=n,positive=pos,negative=n-pos,majority_accuracy=max(pos,n-pos)/n,all_losing_states=sum(not any(data[i]["labels"]) for i in ids),all_winning_states=sum(all(data[i]["labels"]) for i in ids),mixed_states=sum(any(data[i]["labels"]) and not all(data[i]["labels"]) for i in ids))
out=dict(train_state_indices=[i for i in range(len(data)) if i not in valid],val_state_indices=sorted(valid),train=summary([i for i in range(len(data)) if i not in valid]),val=summary(sorted(valid)),regenerated_draws=attempts,excluded182_overlap=0,independent_root_rechecks=20,all_action_sets_legal=True,timeout_note="Original individual timeout ledger was not saved; RNG-replayed skipped draws are not an independently timed retry.",pretrain_sha256=hashlib.sha256((R/"pretrain.json.gz").read_bytes()).hexdigest())
(R/"audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if not k.endswith("indices")}))
