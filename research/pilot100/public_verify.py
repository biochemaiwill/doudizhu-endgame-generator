"""Verify explicit publicly described three-player positions without trusting their answers."""
from build import *
from endgame.fast import hand_from_text,move_from_text
def main():
 seeds=json.loads((ROOT/"public-inputs.json").read_text())
 out=[]
 for sid,hands,turn,last,passes,url,note in seeds:
  row={"source_id":sid,"source_url":url,"source_note":note,"hands":hands,"turn":turn,"last_text":last,"passes":passes}
  try:
   pos=Position(tuple(hand_from_text(h) for h in hands),turn,move_from_text(last) if last else None,passes)
   validate_position(pos)
   row.update(key=state_key(pos),last=move_record(pos.last) if pos.last else None,n_cards=sum(map(hand_size,pos.hands)))
   signal.setitimer(signal.ITIMER_REAL,12)
   solver=Solver(node_limit=900000)
   vals=solver.move_values(pos)
   row.update(moves=[move_record(m,w) for m,w in vals],winning_moves=[m.text() for m,w in vals if w])
   if not row["winning_moves"]:row["status"]="verified_losing_at_requested_turn"
   else:
    signal.setitimer(signal.ITIMER_REAL,20)
    ref_cached.cache_clear()
    rs=sorted((kind,tuple(cards),bool(w)) for ((kind,cards),w) in reference_values(pos))
    if rs!=signature(vals):raise AssertionError("reference mismatch")
    row["reference_states"]=ref_cached.cache_info().currsize
    row["pv"]=trace(pos,solver)
    row["trap_examples"]=[{"first":m.text(),"continuation":trace(pos.play(m),solver)} for m,w in vals if not w][:3]
    row["status"]="verified";row["source"]="public_position"
  except (TimeBudget,BudgetExceeded) as e:row["status"]="unknown_budget";row["budget_reason"]=type(e).__name__
  except Exception as e:row["status"]="error";row["error"]=repr(e)
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  out.append(row);(ROOT/"public-results.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
  print(sid,row["status"],row.get("winning_moves"),row.get("error",""),flush=True)
if __name__=="__main__":main()
