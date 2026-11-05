from pathlib import Path
import sys,json,signal,hashlib
from collections import Counter,deque
B=Path("/home/wutianxiang/doudizhu-endgame-generator");O=B/"research/mechanism_pairs_v2"
sys.path.insert(0,str(B/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
def certify(p):
 validate_position(p);signal.setitimer(signal.ITIMER_REAL,20)
 try:
  s=Solver(node_limit=2000000);v=s.move_values(p)
  assert signature(v)==signature(Solver(node_limit=2000000,canonical=False).move_values(p))
  assert ref.move_values(p)==[((t,c),w) for t,c,w in signature(v)]
  return s,v
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def play(p,names):
 for name in names:
  m=next(m for m in p.legal() if m.text()==name);assert not p.winner_after(m);p=p.play(m)
 return p
def proof(p):
 s,v=certify(p);win=[m for m,w in v if w];assert win and not any(p.winner_after(m) for m in p.legal())
 queue=deque([(p,[])]);seen=set();side=p.side(p.turn)
 while queue and len(seen)<10000:
  q,path=queue.popleft();key=(q.hands,q.turn,q.last,q.passes)
  if key in seen:continue
  seen.add(key);vv=s.move_values(q);bad=[m for m,w in vv if not w]
  if q.side(q.turn)==side and any(w for m,w in vv) and bad:break
  if len(path)>=12:continue
  for m,w in vv:
   if q.side(q.turn)==side and not w:continue
   if not q.winner_after(m):queue.append((q.play(m),path+[m.text()]))
 else:raise ValueError("no verified reachable mistake")
 certify(q);m=bad[0]
 return dict(root=[move_record(m,w) for m,w in v],good_first=win[0].text(),good_line=trace(p.play(win[0]),s,70),bad_prefix=path,bad_action=m.text(),bad_line=trace(q.play(m),s,70),key=state_key(p))
def main():
 candidates=json.loads((O/"candidate_pairs_deep.json").read_text());rows=[];pairs=[];evidence={}
 labels=[
 ("控制牌使用顺序",3,2,"必须先8而不是先33、AA或小王，之后还要保留高对处理农民的配合。","交换大小王后可先33或AA，保留最高单张控制即可，开局不再限制为8。"),
 ("延迟兑现高对子",3,2,"先4或5，AA与小王不可提前出；需要先处理两个低单再兑现对子控制。","换得大王后能分开使用双王控制低单，避免王炸一次消耗两个控制张。"),
 ("跟牌保型与接力",3,2,"面对8只能K或A，拆J、Q或过牌均败；需保留QQQ与对子用于后续清牌。","大王转入队友后可过或用J/Q/K/A接，允许普通三张带牌与对子清理。"),
 ("拆高对与连续控制",3,2,"只有先拆AA出A，随后另一A和小王的顺序仍受限制，整对AA或88先出均败。","地主炸弹改成三张后，先88或AA均能保胜，普通对子清理和小王回收即可。")]
 oldkeys=set()
 for f in [B/"research/scorer_v4/dataset.jsonl",B/"research/grade4_expansion_v5/dataset.jsonl",B/"research/mechanism_pairs_v1/dataset.jsonl"]:
  oldkeys|={x["verification"]["key"] for x in map(json.loads,f.read_text().splitlines())}
 for i,(item,lab) in enumerate(zip(candidates,labels),1):
  if i==2:continue # independent review: ordinary control problem, no established 3/2 gap
  mechanism,hg,eg,hr,er=lab;ids=[]
  for side,grade,reason in [("hard",hg,hr),("easy",eg,er)]:
   source=item[side];p=restore(source);ev=proof(p);assert ev["key"] not in oldkeys;oldkeys.add(ev["key"])
   ident=f"MP2-{i:02d}-{side}";ids.append(ident);evidence[ident]=ev
   inp=dict(hands=[hand_to_cards(h) for h in p.hands],turn=p.turn,last=dict(cards=list(p.last.cards),type=int(p.last.type),main=p.last.main,length=p.last.length) if p.last else None,passes=p.passes)
   rows.append(dict(id=ident,input=inp,target=dict(overall=grade,reasoning_depth=grade,move_concealment=grade),reason=reason,mechanism=mechanism,annotation=dict(type="assistant_provisional_estimate",rubric="v0.1",human_calibrated=False),provenance=dict(source="previously_unlabelled_synthetic_pool" if side=="hard" else "new_swap_counterfactual",origin_id=item["hard"]["origin_id"],family=item["hard"]["family"],swap=item["swap"] if side=="easy" else None),verification=dict(status="verified",key=ev["key"],evidence="mechanism_evidence.json")))
  a,b=rows[-2:];assert [len(h) for h in a["input"]["hands"]]==[len(h) for h in b["input"]["hands"]];assert Counter(sum(a["input"]["hands"],[]))==Counter(sum(b["input"]["hands"],[]))
  assert all(a["input"][k]==b["input"][k] for k in ["turn","last","passes"])
  pairs.append(dict(id=f"MP2P{i:02d}",hard_id=ids[0],easy_id=ids[1],mechanism=mechanism))
 for name,obj in [("pairs.json",pairs),("mechanism_evidence.json",evidence)]:(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 (O/"dataset.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
 print("BUILT",len(rows),len(pairs))
if __name__=="__main__":main()
