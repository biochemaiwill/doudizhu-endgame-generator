"""Reverify fetched source positions using independent short-circuit minimax."""
from build import *
from endgame.fast import hand_from_text,move_from_text,gen_all
import reference_ordered as reference_fast
def restore(row):
 last=None
 if row.get("last_text"):
  last=move_from_text(row["last_text"])
 elif row.get("last"):
  v=row["last"];matches=[m for m in gen_all(hand_from_cards(v["cards"])) if m.cards==tuple(v["cards"]) and m.type==v["type"]]
  if len(matches)!=1:raise ValueError("ambiguous last")
  last=matches[0]
 return Position(tuple(hand_from_text(h) for h in row["hands"]),row["turn"],last,row["passes"])
def run():
 public=[]
 for sid,hands,turn,last,passes,url,note in json.loads((ROOT/"public-inputs.json").read_text()):
  public.append(dict(source="public_position",source_id=sid,hands=hands,turn=turn,last_text=last,passes=passes,source_url=url,source_note=note))
 master=[]
 out=[];counts=Counter();seen=set()
 for i,r in enumerate(public+master):
  row=dict(r)
  try:
   p=restore(row);validate_position(p);key=state_key(p)
   row.update(key=key,n_cards=sum(map(hand_size,p.hands)),last=move_record(p.last) if p.last else None)
   if key in seen:row["status"]="duplicate"
   elif row["source"]!="public_position" and row["n_cards"]>18:row["status"]="size_limit"
   else:
    seen.add(key);signal.setitimer(signal.ITIMER_REAL,18 if i<len(public) else 4)
    s=Solver(node_limit=700000);vals=s.move_values(p)
    row["moves"]=[move_record(m,w) for m,w in vals]
    row["winning_moves"]=[m.text() for m,w in vals if w]
    if not row["winning_moves"]:row["status"]="losing"
    elif len(vals)==1:row["status"]="forced_no_choice"
    else:
     reference_fast.solve.cache_clear()
     rv=reference_fast.move_values(p)
     sig=[((kind,cards),w) for kind,cards,w in signature(vals)]
     if rv!=sig:raise AssertionError("independent mismatch")
     if signature(Solver(node_limit=700000,canonical=False).move_values(p))!=signature(vals):raise AssertionError("canonical mismatch")
     row["reference_states"]=reference_fast.solve.cache_info().currsize
     row["pv"]=trace(p,s)
     row["trap_examples"]=[{"first":m.text(),"continuation":trace(p.play(m),s)} for m,w in vals if not w][:2]
     row["status"]="verified"
  except (TimeBudget,BudgetExceeded) as e:row["status"]="unknown_budget";row["reason"]=type(e).__name__
  except Exception as e:row["status"]="error";row["error"]=repr(e)
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  out.append(row);counts[row["status"]]+=1
  (ROOT/"public-ordered-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
  if i<len(public) or i%30==0:print(i,row.get("source_id",row.get("source_row")),row["status"],row.get("winning_moves"),dict(counts),flush=True)
 (ROOT/"public-ordered-run.json").write_text(json.dumps(dict(counts),indent=2),encoding="utf-8")
 print("SOURCE_DONE",dict(counts),flush=True)
if __name__=="__main__":run()
