from pathlib import Path
import json
d=Path(__file__).resolve().parent
r=json.loads((d/"result.json").read_text())
assert r["status"]=="verified"
assert len(r["modes"])==3
signatures=[x["root_signature"] for x in r["modes"]]
assert signatures[0]==signatures[1]==signatures[2]
assert len(signatures[0])==34
assert sum(v for t,c,v in signatures[0])==1
assert r["winning_moves"]==["2"]
branches=json.loads((d/"after_joker_defense.json").read_text())["branches"]
for rank in ("7","9"):
 b=next(b for b in branches if b["farmer_down_lead"]==rank)
 assert b["landlord_wins"]==["Q"]
assert json.loads((d/"annotation.json").read_text())["target"]=={"overall":4,"reasoning_depth":4,"move_concealment":4}
print("PASS: three identical 34-action signatures, only 2 wins; key 7/9 defenses require Q; 4/4/4 label")
