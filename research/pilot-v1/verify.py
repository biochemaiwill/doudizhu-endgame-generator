"""Reproduce six-case pilot verification. No model/API calls or training."""
import sys, json, time, platform, importlib.util
from pathlib import Path
from collections import Counter
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"vendor"))
from endgame.puzzles import PUZZLES, Puzzle
from endgame.solver import Solver
from endgame.fast import parse_cards
spec=importlib.util.spec_from_file_location("pilot_reference",ROOT/"vendor/tests/reference_solver.py")
ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
original=ref.ref_landlord_wins
@lru_cache(maxsize=None)
def cached_ref(hands,turn,last,passes):
    return original([Counter(dict(h)) for h in hands],turn,last,passes)
def wrapper(hands,turn=0,last=None,passes=0):
    return cached_ref(tuple(tuple(sorted(h.items())) for h in hands),turn,last,passes)
ref.ref_landlord_wins=wrapper
cases=[
 Puzzle("control_single","control",("3","4","5"),0,"3",""),
 Puzzle("control_pair","control",("2 2 3","4 4","5 5"),0,"2 2",""),
 *[p for p in PUZZLES if len(p.hands)==3]
]
out={"label_status":"pilot_only_not_human_calibrated","python":platform.python_version(),"source_commit":json.loads((ROOT/"source-manifest.json").read_text())["commit"],"cases":[]}
for p in cases:
    started=time.monotonic(); cached_ref.cache_clear()
    hands=[Counter(parse_cards(h)) for h in p.hands]
    total=sum(hands,Counter())
    assert all(c <= (1 if r>=16 else 4) for r,c in total.items())
    pos=p.position()
    assert pos.last is None
    normal=Solver(node_limit=500000)
    moves=normal.move_values(pos)
    plain=Solver(node_limit=500000,canonical=False).move_values(pos)
    def signature(rows):
        return sorted((int(m.type),tuple(m.cards),bool(w)) for m,w in rows)
    independent=ref.ref_move_values(hands,p.turn,None,p.passes)
    ref_sig=sorted((kind,tuple(cards),bool(w)) for ((kind,cards),w) in independent)
    assert signature(moves)==signature(plain)==ref_sig, p.key
    assert any(w for _,w in moves)
    evidence=[]
    for m,w in moves:
        row={"move":m.text(),"type":int(m.type),"mover_forced_win":bool(w)}
        if not pos.winner_after(m):
            after=pos.play(m)
            row["next_player"]=after.turn
            row["replies"]=[{"move":x.text(),"next_mover_forced_win":bool(v)} for x,v in Solver(node_limit=500000).move_values(after)]
        evidence.append(row)
    case={"id":p.key,"hands":list(p.hands),"turn":p.turn,"last":None,"passes":0,"n_cards":sum(map(len,[list(parse_cards(h)) for h in p.hands])),"legal_moves":len(moves),"winning_moves":[m.text() for m,w in moves if w],"three_checks_agree":True,"independent_cache_states":cached_ref.cache_info().currsize,"seconds":round(time.monotonic()-started,4),"root_move_evidence":evidence}
    out["cases"].append(case)
    (ROOT/"verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in case.items() if k!="root_move_evidence"},ensure_ascii=False),flush=True)
print("PASS: all legal root move outcomes matched optimized solver, uncanonicalized solver and independent reference.")
