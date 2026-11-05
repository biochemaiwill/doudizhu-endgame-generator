# -*- coding: utf-8 -*-
"""独立的"笨"参考求解器：只用 doudizhu/moves.py（doudizhu-dmc-lab 项目复制来的原始规则引擎）+ 朴素极小极大，
不用置换表、不用排序、不用规范化、不用整数编码。只适合极小的残局（每家 ≤4 张），
用来和 endgame/solver.py 的快速求解器逐局对答案。"""
from collections import Counter

from doudizhu.moves import gen_legal, MoveType


def ref_landlord_wins(hands, turn=0, last=None, passes=0):
    """hands: list[Counter]，0 号为地主。last: doudizhu.moves.Move 或 None。"""
    n = len(hands)
    hand = hands[turn]
    results = []
    for m in gen_legal(hand, last):
        if m.type == MoveType.PASS:
            if passes + 1 >= n - 1:
                v = ref_landlord_wins(hands, (turn + 1) % n, None, 0)
            else:
                v = ref_landlord_wins(hands, (turn + 1) % n, last, passes + 1)
        else:
            nh = [h.copy() for h in hands]
            nh[turn].subtract(m.cards)
            nh[turn] = +nh[turn]
            if not nh[turn]:
                v = (turn == 0)
            else:
                v = ref_landlord_wins(nh, (turn + 1) % n, m, 0)
        results.append(v)
    return any(results) if turn == 0 else all(results)


def ref_move_values(hands, turn=0, last=None, passes=0):
    """返回 [(cards 元组, 行动方是否必胜)]。"""
    n = len(hands)
    out = []
    for m in gen_legal(hands[turn], last):
        if m.type == MoveType.PASS:
            if passes + 1 >= n - 1:
                v = ref_landlord_wins(hands, (turn + 1) % n, None, 0)
            else:
                v = ref_landlord_wins(hands, (turn + 1) % n, last, passes + 1)
        else:
            nh = [h.copy() for h in hands]
            nh[turn].subtract(m.cards)
            nh[turn] = +nh[turn]
            v = (turn == 0) if not nh[turn] else ref_landlord_wins(nh, (turn + 1) % n, m, 0)
        out.append(((int(m.type), m.cards), v == (turn == 0)))
    return out
