"""Build a four-mechanism evidence pilot; does not train or change old data."""
from pathlib import Path
import sys,json,signal,hashlib
from collections import Counter,deque
B=Path("/home/wutianxiang/doudizhu-endgame-generator");OUT=B/"research/mechanism_pairs_v1"
sys.path.insert(0,str(B/"research/pilot100"))
from build import *
import reference_ordered as ref
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def pos(r):return Position(tuple(hand_from_cards(h) for h in r["input"]["hands"]),r["input"]["turn"])
def replay(p,names):
 steps=[]
 for text in names:
  matches=[m for m in p.legal() if m.text()==text]
  assert len(matches)==1,(text,p.describe())
  m=matches[0];steps.append({"player":p.turn,**move_record(m)})
  assert not p.winner_after(m),"prefix must not run past terminal"
  p=p.play(m)
 return p,steps
def certify(p):
 validate_position(p);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  s=Solver(node_limit=2000000);vals=s.move_values(p);sig=signature(vals)
  assert sig==signature(Solver(node_limit=2000000,canonical=False).move_values(p))
  assert ref.move_values(p)==[((t,c),w) for t,c,w in sig]
  return s,vals
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def checked_line(p,line):
 for i,step in enumerate(line):
  assert step["player"]==p.turn
  m=next(m for m in p.legal() if list(m.cards)==step["cards"] and int(m.type)==step["type"])
  winner=p.winner_after(m)
  if winner:
   assert i==len(line)-1
   return winner
  p=p.play(m)
 raise AssertionError("nonterminal stored trace")
def main():
 old={}
 for path in [B/"research/scorer_v4/dataset.jsonl",B/"research/grade4_expansion_v5/dataset.jsonl"]:
  old.update({r["id"]:r for r in map(json.loads,path.read_text().splitlines())})
 rows={i:dict(old[i]) for i in ["EX222","NX325","EX211","NX323","EX208","NX322","NX313"]}
 t=json.loads((OUT/"carry_trials.json").read_text())[0]
 new=dict(id="MPNEW001",input=dict(old["NX313"]["input"],hands=t["hands"]),target={"overall":2,"reasoning_depth":2,"move_concealment":2},
  reason="可直接炸8888后用1010清理对子，保留大王回收单张；仍不能把大王与10作为四带二过早送掉。比原题限定9J或9A的带牌选择更宽松。",annotation={"type":"assistant_provisional_estimate","rubric":"v0.1","human_calibrated":False,"blind":False},
  provenance={"source":"new_counterfactual_swap","origin_id":t["origin"],"parent_id":"NX313","family":old["NX313"]["provenance"]["family"],"mutation":{"players":[0,2],"cards":[14,17]}})
 assert state_key(pos(new)) not in {r["verification"]["key"] for r in old.values()}
 rows[new["id"]]=new
 proof={}
 for ident,r in rows.items():
  p=pos(r);s,vals=certify(p)
  wins=[m for m,w in vals if w];losses=[m for m,w in vals if not w]
  assert wins
  assert not any(p.winner_after(m) for m in p.legal())
  winning=trace(p.play(wins[0]),s,maxplies=70)
  assert checked_line(p.play(wins[0]),winning)==p.side(p.turn)
  q=p;prefix=[];wrong=None
  for step in trace(p,s,maxplies=70):
   vv=s.move_values(q);bad=[m for m,w in vv if not w]
   if q.side(q.turn)==p.side(p.turn) and bad:
    wrong=bad[0];break
   m=next(m for m in q.legal() if m.text()==step["move"])
   if q.winner_after(m):break
   prefix.append(m.text());q=q.play(m)
  if wrong is None:
   queue=deque([(p,[])]);seen=set()
   while queue and len(seen)<5000:
    qq,path=queue.popleft();kk=(qq.hands,qq.turn,qq.last,qq.passes)
    if kk in seen:continue
    seen.add(kk);vv=s.move_values(qq)
    bb=[m for m,w in vv if not w]
    if qq.side(qq.turn)==p.side(p.turn) and bb:q,prefix,wrong=qq,path,bb[0];break
    if len(path)>=14:continue
    for mm,ww in vv:
     if qq.side(qq.turn)==p.side(p.turn) and not ww:continue
     if not qq.winner_after(mm):queue.append((qq.play(mm),path+[mm.text()]))
  assert wrong is not None,"Each included example must have a reachable losing choice"
  certify(q)
  losing=trace(q.play(wrong),s,maxplies=70)
  if wrong:assert checked_line(q.play(wrong),losing)!=p.side(p.turn)
  proof[ident]={"status":"three_way_pass","key":state_key(p),"root_outcomes":[move_record(m,w) for m,w in vals],
   "example_winning_first":wins[0].text(),"winning_continuation":winning,
   "wrong_prefix":prefix if wrong else None,"example_wrong_action":wrong.text() if wrong else None,"wrong_continuation":losing,
   "note":"Example continuation is illustrative; all-root outcomes are exact. If all root moves win, a later reachable mistake is recorded when found; never invent a losing root move."}
  r["pilot_provenance"]={"reused_existing":ident!="MPNEW001","original_id":ident if ident!="MPNEW001" else None}
  r["verification"]={"status":"verified","key":state_key(p),"winning_moves":[m.text() for m in wins],"evidence":"mechanism_evidence.json"}
 facts=[
  ("SPLIT","拆对子","EX222","NX325",["8","过","过","Q","过","过"],["3","4"],["3 3","4 4","2","小王"],"先8再Q后，当前手牌33、44必须拆成单张；整对出或先耗控制牌均败。"),
  ("TEAM","让牌／队友接管","EX211","NX323",["10 10","过"],["J J"],["过"],"先1010后，队友必须用JJ接管；此处谦让过牌反而使本方输。"),
  ("CARRY","带牌选择","NX313","MPNEW001",[],["8 8 8 8 9 J","8 8 8 8 9 A"],["8 8 8 8","8 8 8 8 10 10"],"四带二只能带9J或9A；直接炸或带1010破坏后续结构。"),
  ("CONTROL","保留控制权／次序","EX208","NX322",[],["9"],["7 7","K K","小王"],"必须先9，不能提前兑现对子或小王；需要保留之后的控制与拆牌次序。")
 ]
 pairs=[];mechanisms=[]
 for num,(code,label,hard,easy,prefix,good,bad,claim) in enumerate(facts,1):
  hp,ep=pos(rows[hard]),pos(rows[easy])
  assert [hand_size(h) for h in hp.hands]==[hand_size(h) for h in ep.hands]
  assert Counter(sum(rows[hard]["input"]["hands"],[]))==Counter(sum(rows[easy]["input"]["hands"],[]))
  assert rows[hard]["provenance"]["family"]==rows[easy]["provenance"]["family"]
  q,steps=replay(hp,prefix);_,out=certify(q);truth={m.text():w for m,w in out}
  for text in good:assert truth[text],(code,text)
  for text in bad:assert not truth[text],(code,text)
  assert sorted(k for k,v in truth.items() if v)==sorted(good)
  goodmove=next(m for m in q.legal() if m.text()==good[0])
  badmove=next(m for m in q.legal() if m.text()==bad[0])
  solver=Solver(node_limit=2000000)
  correctline=trace(q.play(goodmove),solver,maxplies=70)
  badline=trace(q.play(badmove),solver,maxplies=70)
  assert checked_line(q.play(goodmove),correctline)==q.side(q.turn)
  assert checked_line(q.play(badmove),badline)!=q.side(q.turn)
  mechanisms.append({"correct_action":good[0],"correct_continuation":correctline,"wrong_action":bad[0],"wrong_continuation":badline,"pair_id":code,"mechanism":label,"claim":claim,"hard_id":hard,"prefix_text":prefix,"prefix":steps,"actor":q.turn,"exact_winning_actions":good,"checked_losing_actions":bad,"all_outcomes":[move_record(m,w) for m,w in out]})
  pairs.append({"id":code,"mechanism":label,"hard_id":hard,"easy_id":easy,"hard_grade":rows[hard]["target"]["overall"],"easy_grade":rows[easy]["target"]["overall"],
    "same_counts":True,"same_global_rank_multiset":True,"both_require_multiple_moves":True,"family":rows[hard]["provenance"]["family"],"difficulty_status":"assistant_estimate_not_human_validated"})
 (OUT/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows.values()))
 save("pairs.json",pairs);save("mechanism_evidence.json",{"positions":proof,"mechanism_facts":mechanisms})
 save("validation.json",{"rows":len(rows),"pairs":len(pairs),"new_states":1,"reused_states":7,"mechanisms":[r["mechanism"] for r in pairs],"all_roots_three_way":True,"all_mechanism_claims_exactly_checked":True,"all_continuations_legally_replayed":True,"no_training_change":True,"dataset_sha256":hashlib.sha256((OUT/"dataset.jsonl").read_bytes()).hexdigest()})
 print("DONE",len(rows),len(pairs),flush=True)
if __name__=="__main__":main()
