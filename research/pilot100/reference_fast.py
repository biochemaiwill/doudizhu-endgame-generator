"""Independent rule-enumerator minimax, short-circuiting exact Boolean outcomes."""
from functools import lru_cache
from collections import Counter
from doudizhu.moves import gen_legal,MoveType,Move
from endgame.fast import hand_to_cards
@lru_cache(maxsize=300000)
def solve(hands,turn,last,passes):
 n=len(hands);mine=turn==0
 counters=[Counter(dict(h)) for h in hands]
 for m in gen_legal(counters[turn],last):
  if m.type==MoveType.PASS:
   v=solve(hands,(turn+1)%n,None if passes+1>=n-1 else last,0 if passes+1>=n-1 else passes+1)
  else:
   rest=counters[turn]-Counter(m.cards)
   if not rest:v=mine
   else:
    updated=list(hands);updated[turn]=tuple(sorted(rest.items()))
    v=solve(tuple(updated),(turn+1)%n,m,0)
  if v==mine:return mine
 return not mine
def move_values(pos):
 solve.cache_clear()
 hands=tuple(tuple(sorted(Counter(hand_to_cards(h)).items())) for h in pos.hands)
 last=Move(MoveType(pos.last.type),pos.last.main,pos.last.length,pos.last.cards) if pos.last else None
 out=[]
 for m in gen_legal(Counter(dict(hands[pos.turn])),last):
  if m.type==MoveType.PASS:
   v=solve(hands,(pos.turn+1)%3,None if pos.passes+1>=2 else last,0 if pos.passes+1>=2 else pos.passes+1)
  else:
   rest=Counter(dict(hands[pos.turn]))-Counter(m.cards)
   if not rest:v=pos.turn==0
   else:
    upd=list(hands);upd[pos.turn]=tuple(sorted(rest.items()))
    v=solve(tuple(upd),(pos.turn+1)%3,m,0)
  out.append(((int(m.type),tuple(m.cards)),v==(pos.turn==0)))
 return sorted(out)
