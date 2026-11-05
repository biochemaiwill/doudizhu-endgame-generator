"""Enrich narrow, non-obvious candidate states; filters are NOT difficulty labels."""
from build import *
from endgame.fast import hand_from_text,gen_all
def run_hard():
 path=ROOT/"hard-candidates.jsonl"
 if path.exists():raise SystemExit("Refusing overwrite")
 existing=[json.loads(x) for x in (ROOT/"candidates.jsonl").read_text().splitlines()]
 seen={r["key"] for r in existing if "key" in r}
 anchor=json.loads((ROOT.parent/"pilot-v1/verification.json").read_text())["cases"]
 for a in anchor:seen.add(state_key(Position(tuple(hand_from_text(h) for h in a["hands"]),a["turn"])))
 rng=random.Random(20261106);start=time.monotonic();counts=Counter();pool=[]
 with path.open("w",encoding="utf-8") as f:
  for i in range(1200):
   if len(pool)>=28 or time.monotonic()-start>540:break
   # Diverse ranks, slightly larger hands, occasional replay prefix.
   ranks=sorted(rng.sample(range(3,16),rng.randint(6,9)))
   deck=[r for r in full_deck() if r in ranks or r>=16];rng.shuffle(deck)
   sizes=[rng.randint(3,7) for _ in range(3)]
   while sum(sizes)>18:sizes[rng.randrange(3)]=rng.randint(3,6)
   k=0;hs=[]
   for n in sizes:hs.append(hand_from_cards(deck[k:k+n]));k+=n
   pos=Position(tuple(hs),rng.randrange(3));history=[]
   if i%3==0:
    for _ in range(rng.randint(1,3)):
     ms=[m for m in pos.legal() if not pos.winner_after(m)]
     if not ms:break
     m=rng.choice(ms);history.append({"player":pos.turn,**move_record(m)});pos=pos.play(m)
   row={"attempt":i,"history":history,"source":"narrow_nonobvious_enrichment"}
   try:
    validate_position(pos);key=state_key(pos)
    row.update(key=key,hands=[hand_to_text(h) for h in pos.hands],turn=pos.turn,last=move_record(pos.last) if pos.last else None,passes=pos.passes,n_cards=sum(map(hand_size,pos.hands)))
    if key in seen:row["status"]="duplicate"
    else:
     seen.add(key);signal.setitimer(signal.ITIMER_REAL,3)
     solver=Solver(node_limit=200000)
     vals=solver.move_values(pos)
     wins=[m for m,w in vals if w]
     row["moves"]=[move_record(m,w) for m,w in vals];row["winning_moves"]=[m.text() for m in wins]
     if not wins:row["status"]="losing"
     elif len(wins)>2 or len(vals)<3 or any(pos.winner_after(m) for m in wins):row["status"]="not_narrow"
     else:
      # Exclude states solved by both the largest-group and smallest-single heuristics.
      legal=[m for m,w in vals if not m.is_pass]
      maxmove=max(legal,key=lambda m:(m.n,-m.main)) if legal else vals[0][0]
      singles=[m for m in legal if m.n==1]
      low=min(singles,key=lambda m:m.main) if singles else maxmove
      if maxmove in wins and low in wins:row["status"]="simple_heuristics_win"
      else:
       signal.setitimer(signal.ITIMER_REAL,8)
       ref_cached.cache_clear()
       rv=reference_values(pos)
       rs=sorted((kind,tuple(cards),bool(w)) for ((kind,cards),w) in rv)
       if rs!=signature(vals):raise AssertionError("reference mismatch")
       if signature(Solver(node_limit=400000,canonical=False).move_values(pos))!=signature(vals):raise AssertionError("canonical mismatch")
       row["reference_states"]=ref_cached.cache_info().currsize
       row["pv"]=trace(pos,solver)
       row["trap_examples"]=[{"first":m.text(),"continuation":trace(pos.play(m),solver)} for m,w in vals if not w][:2]
       row["status"]="verified";pool.append(row)
   except (TimeBudget,BudgetExceeded) as e:row["status"]="timeout" if isinstance(e,TimeBudget) else "node_budget"
   except Exception as e:row["status"]="error";row["error"]=repr(e)
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   counts[row["status"]]+=1;f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
   if i%50==0:print({"attempted":i+1,"verified":len(pool),"counts":dict(counts),"seconds":round(time.monotonic()-start)},flush=True)
 (ROOT/"hard-pool.json").write_text(json.dumps(pool,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (ROOT/"hard-run.json").write_text(json.dumps({"seed":20261106,"counts":dict(counts),"verified":len(pool),"seconds":time.monotonic()-start,"filters":"at most two winning root actions; at least three choices; no immediate win; largest-group and smallest-single not both winning","label_status":"unlabeled; filter is not difficulty"},indent=2),encoding="utf-8")
 print("HARD_FINISHED",dict(counts),len(pool),flush=True)
if __name__=="__main__":run_hard()
