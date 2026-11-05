"""Bounded synthetic three-player pilot builder. Python standard library only."""
import sys,json,random,time,signal,platform,hashlib,importlib.util,argparse
from pathlib import Path
from collections import Counter,defaultdict,deque
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
VENDOR=ROOT.parent/"pilot-v1/vendor"
sys.path.insert(0,str(VENDOR))
from endgame.solver import Position,Solver,BudgetExceeded,_canon
from endgame.fast import hand_from_cards,hand_to_cards,hand_to_text,hand_size
from doudizhu.cards import full_deck
from doudizhu.moves import Move,MoveType
spec=importlib.util.spec_from_file_location("reference",VENDOR/"tests/reference_solver.py")
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
original=reference.ref_landlord_wins
@lru_cache(maxsize=300000)
def ref_cached(hands,turn,last,passes):
 return original([Counter(dict(h)) for h in hands],turn,last,passes)
def ref_wrap(hands,turn=0,last=None,passes=0):
 return ref_cached(tuple(tuple(sorted(h.items())) for h in hands),turn,last,passes)
reference.ref_landlord_wins=ref_wrap
class TimeBudget(Exception): pass
def alarm(*_): raise TimeBudget()
signal.signal(signal.SIGALRM,alarm)
def validate_position(pos):
 if pos.n!=3 or pos.turn not in (0,1,2): raise ValueError("three players/turn")
 if any(hand_size(h)==0 for h in pos.hands): raise ValueError("terminal")
 counts=Counter(r for h in pos.hands for r in hand_to_cards(h))
 if pos.last:
  counts.update(pos.last.cards)
 if any(r<3 or r>17 or c>(1 if r>=16 else 4) for r,c in counts.items()): raise ValueError("physical cards")
 if not 0<=pos.passes<=1 or (pos.last is None and pos.passes!=0): raise ValueError("passes")
 return True
def state_key(pos):
 hs,lk=_canon(pos.hands,pos.last.key if pos.last else None)
 return hashlib.sha256(repr((hs,lk,pos.turn,pos.passes)).encode()).hexdigest()
def eligible(row): return row.get("status")=="verified" and bool(row.get("winning_moves"))
def move_record(m,w=None):
 out={"cards":list(m.cards),"move":m.text(),"type":int(m.type)}
 if w is not None: out["mover_wins"]=bool(w)
 return out
def signature(rows): return sorted((int(m.type),tuple(m.cards),bool(w)) for m,w in rows)
def reference_values(pos):
 last=Move(MoveType(pos.last.type),pos.last.main,pos.last.length,pos.last.cards) if pos.last else None
 return reference.ref_move_values([Counter(hand_to_cards(h)) for h in pos.hands],pos.turn,last,pos.passes)
def sample(rng,index):
 # Mix full-deck ranks and locally concentrated ranks to vary combinations.
 ranks=list(range(3,16))
 if index%3:
  ranks=sorted(rng.sample(ranks,rng.randint(5,8)))
 deck=[r for r in full_deck() if r in ranks or r>=16]
 rng.shuffle(deck)
 sizes=[rng.randint(2,6) for _ in range(3)]
 while sum(sizes)>16: sizes[rng.randrange(3)]=rng.randint(2,5)
 hands=[];i=0
 for n in sizes: hands.append(hand_from_cards(deck[i:i+n]));i+=n
 pos=Position(tuple(hands),rng.randrange(3))
 history=[]
 if index%4==0:
  for _ in range(rng.randint(1,3)):
   choices=[m for m in pos.legal() if not pos.winner_after(m)]
   if not choices: break
   m=rng.choice(choices);history.append({"player":pos.turn,**move_record(m)})
   pos=pos.play(m)
 return pos,history
def trace(pos,solver,maxplies=24):
 out=[]
 for _ in range(maxplies):
  vals=solver.move_values(pos)
  wins=[m for m,w in vals if w]
  # Solver arbitrary loss fallback: demonstration only, not human-optimal defense.
  m=(wins or [m for m,w in vals])[0]
  out.append({"player":pos.turn,"legal":len(vals),"winning":len(wins),**move_record(m)})
  if pos.winner_after(m):break
  pos=pos.play(m)
 return out
def run():
 parser=argparse.ArgumentParser();parser.add_argument("--pool",type=int,default=180);args=parser.parse_args()
 target=ROOT/"candidates.jsonl"
 if target.exists(): raise SystemExit("Refusing to overwrite evidence")
 rng=random.Random(20261105)
 anchors=json.loads((ROOT.parent/"pilot-v1/verification.json").read_text())["cases"]
 blocked=set()
 for a in anchors: blocked.add(state_key(Position(tuple(hand_from_cards(__import__("endgame.fast",fromlist=["parse_cards"]).parse_cards(h)) for h in a["hands"]),a["turn"])))
 seen=set(blocked);pool=[];start=time.monotonic();counts=Counter()
 with target.open("w",encoding="utf-8") as f:
  for i in range(1600):
   if len(pool)>=args.pool or time.monotonic()-start>1500:break
   pos,hist=sample(rng,i);row={"attempt":i,"history":hist}
   try:
    validate_position(pos);key=state_key(pos)
    row.update(key=key,hands=[hand_to_text(h) for h in pos.hands],turn=pos.turn,last=move_record(pos.last) if pos.last else None,passes=pos.passes,n_cards=sum(map(hand_size,pos.hands)))
    if key in seen: row["status"]="duplicate"
    else:
     seen.add(key)
     signal.setitimer(signal.ITIMER_REAL,3.0)
     solver=Solver(node_limit=120000)
     vals=solver.move_values(pos)
     row["moves"]=[move_record(m,w) for m,w in vals]
     row["winning_moves"]=[m.text() for m,w in vals if w]
     if not row["winning_moves"]:row["status"]="losing"
     elif len(vals)<2:row["status"]="forced_no_choice"
     else:
      ref_cached.cache_clear()
      rv=reference_values(pos)
      rs=sorted((kind,tuple(cards),bool(w)) for ((kind,cards),w) in rv)
      if rs!=signature(vals):raise AssertionError("reference mismatch")
      uncanon=Solver(node_limit=120000,canonical=False).move_values(pos)
      if signature(uncanon)!=signature(vals):raise AssertionError("canonical mismatch")
      row["status"]="verified";row["reference_states"]=ref_cached.cache_info().currsize
      row["pv"]=trace(pos,solver)
      traps=[m for m,w in vals if not w]
      row["trap_examples"]=[]
      for m in sorted(traps,key=lambda m:(-m.n,m.main))[:2]:
       row["trap_examples"].append({"first":m.text(),"continuation":trace(pos.play(m),solver)})
      pool.append(row)
   except (TimeBudget,BudgetExceeded) as e: row["status"]="timeout" if isinstance(e,TimeBudget) else "node_budget"
   except Exception as e:
    row["status"]="error";row["error"]=type(e).__name__+": "+str(e)
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   counts[row["status"]]+=1
   f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
   if i%40==0:print(json.dumps({"attempted":i+1,"verified":len(pool),"counts":dict(counts),"seconds":round(time.monotonic()-start)},ensure_ascii=False),flush=True)
 # Only append after all evidence collection succeeds; select deterministically by strata, not difficulty scores.
 pool=[r for r in pool if eligible(r)]
 buckets=defaultdict(deque)
 for r in pool:
  win_types=tuple(sorted(set(m["type"] for m in r["moves"] if m["mover_wins"])))
  ratio=len(r["winning_moves"])/len(r["moves"])
  bucket=(r["turn"],r["last"] is not None,win_types,"all" if ratio==1 else "single" if len(r["winning_moves"])==1 else "multiple")
  buckets[bucket].append(r)
 selected=[]
 while len(selected)<100 and any(buckets.values()):
  for k in sorted(buckets,key=repr):
   if buckets[k] and len(selected)<100:selected.append(buckets[k].popleft())
 for j,r in enumerate(selected,1):r["id"]=f"DZ{j:03d}"
 (ROOT/"selection.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 manifest={"seed":20261105,"python":platform.python_version(),"source_commit":json.loads((ROOT.parent/"pilot-v1/source-manifest.json").read_text())["commit"],"counts":dict(counts),"selected":len(selected),"elapsed_seconds":round(time.monotonic()-start,3),"reference_timeout_seconds":3,"optimized_node_limit":120000,"data_source":"synthetic remaining hands plus optional legal prefix; not full historical game","label_status":"not_yet_annotated"}
 (ROOT/"run.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(manifest),flush=True)
 if len(selected)!=100:raise SystemExit("Incomplete selection")
if __name__=="__main__":run()
