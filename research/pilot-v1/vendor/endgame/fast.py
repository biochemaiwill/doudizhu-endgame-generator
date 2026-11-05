# -*- coding: utf-8 -*-
"""整数编码的快速牌型系统（求解器专用）。

编码：一手牌 = 一个 Python int，每个点数占 4 bit（点数 3..17 → 第 0..14 组）。
  例：两张 3 + 一张大王 = (2 << 0) | (1 << 56)
出牌就是整数减法（hand - move.delta），局面哈希就是几个 int 组成的 tuple。

牌型规则逐条沿用 doudizhu/moves.py（从 doudizhu-dmc-lab 复制来的规则引擎）：
  顺子 ≥5、连对 ≥3 对、飞机 ≥2 组，只能用 3~A；三带一/飞机翅膀不能与三张同点；
  王炸不可拆带；四带两对须两个不同点数的对子；炸弹压一切非炸，王炸最大。
tests/test_solver.py 在上千手随机牌上逐一核对本文件与 doudizhu/moves.py
生成的着法集合完全相同。
"""
import itertools

from doudizhu.cards import CHAR_RANKS, RANK_NAMES, SMALL_JOKER, BIG_JOKER, CHAIN_MAX_RANK
from doudizhu.moves import MoveType, TYPE_NAMES

MAX_KICKER_COMBOS = 200
RANKS = list(range(3, 18))
_SHIFT = {r: 4 * (r - 3) for r in RANKS}
_ONE = {r: 1 << _SHIFT[r] for r in RANKS}

T_PASS = int(MoveType.PASS)
T_BOMB = int(MoveType.BOMB)
T_ROCKET = int(MoveType.ROCKET)


class FMove:
    """一手牌。key=(牌型, 主点数, 链长) 决定能否被压；delta 是要从手牌里减掉的整数。"""
    __slots__ = ("type", "main", "length", "cards", "delta", "n", "key")

    def __init__(self, mtype, main, length, cards):
        self.type = int(mtype)
        self.main = main
        self.length = length
        self.cards = tuple(sorted(cards))
        self.delta = sum(_ONE[r] for r in self.cards)
        self.n = len(self.cards)
        self.key = (self.type, main, length)

    @property
    def is_pass(self):
        return self.type == T_PASS

    @property
    def is_bomb(self):
        return self.type in (T_BOMB, T_ROCKET)

    def text(self):
        """紧凑文字：'3 3 4 4 5 5'；过牌为'过'。"""
        if self.type == T_PASS:
            return "过"
        return " ".join(RANK_NAMES[r] for r in self.cards)

    def label(self):
        """带牌型名的文字：'连对 [3 3 4 4 5 5]'。"""
        if self.type == T_PASS:
            return "过"
        return f"{TYPE_NAMES[MoveType(self.type)]} [{self.text()}]"

    def __repr__(self):
        return f"FMove({self.label()})"

    def __eq__(self, other):
        return isinstance(other, FMove) and self.type == other.type and self.cards == other.cards \
            and self.main == other.main and self.length == other.length

    def __hash__(self):
        return hash((self.type, self.main, self.length, self.cards))


PASS_MOVE = FMove(MoveType.PASS, 0, 0, ())


# ---------------------------------------------------------------- 编解码
def hand_from_cards(cards):
    h = 0
    for r in cards:
        h += _ONE[r]
    return h


def hand_counts(hand):
    """返回 dict 点数→张数（只含 >0 的）。"""
    out = {}
    for r in RANKS:
        c = (hand >> _SHIFT[r]) & 0xF
        if c:
            out[r] = c
    return out


def hand_to_cards(hand):
    out = []
    for r in RANKS:
        c = (hand >> _SHIFT[r]) & 0xF
        out.extend([r] * c)
    return out


def hand_size(hand):
    n = 0
    while hand:
        n += hand & 0xF
        hand >>= 4
    return n


def count_of(hand, r):
    return (hand >> _SHIFT[r]) & 0xF


def parse_cards(text):
    """'33 4 T J Q K A 2 X D' / '10' / '小王' '大王' → 点数列表。"""
    t = str(text).strip().upper()
    t = t.replace("小王", "X").replace("大王", "D").replace("10", "T")
    ranks = []
    for ch in t:
        if ch in " ,，、[]":
            continue
        if ch not in CHAR_RANKS:
            raise ValueError(f"无法识别的牌: {ch}（可用 3-9 T(=10) J Q K A 2 X(小王) D(大王)）")
        ranks.append(CHAR_RANKS[ch])
    return ranks


def hand_from_text(text):
    cards = parse_cards(text)
    h = hand_from_cards(cards)
    for r, c in hand_counts(h).items():
        if c > (1 if r >= 16 else 4):
            raise ValueError(f"点数 {RANK_NAMES[r]} 有 {c} 张，超过一副牌上限")
    return h


def hand_to_text(hand):
    cards = hand_to_cards(hand)
    return " ".join(RANK_NAMES[r] for r in cards) if cards else "（空）"


# ---------------------------------------------------------------- 生成
def _chain_windows(cnt, min_count, min_len, max_len):
    ok = [r for r in range(3, CHAIN_MAX_RANK + 1) if cnt.get(r, 0) >= min_count]
    okset = set(ok)
    out = []
    for start in ok:
        length = 0
        while start + length in okset:
            length += 1
            if min_len <= length <= max_len:
                out.append(tuple(range(start, start + length)))
    return out


def _kicker_singles(cnt, banned, n):
    pool = []
    for r in sorted(cnt):
        if r in banned or cnt[r] <= 0:
            continue
        pool.extend([r] * min(cnt[r], n))
    seen = set()
    for combo in itertools.combinations(pool, n):
        if SMALL_JOKER in combo and BIG_JOKER in combo:
            continue
        if combo not in seen:
            seen.add(combo)
            if len(seen) >= MAX_KICKER_COMBOS:
                break
    return sorted(seen)


def _kicker_pairs(cnt, banned, n):
    avail = [r for r in sorted(cnt) if r not in banned and r <= 15 and cnt[r] >= 2]
    return list(itertools.islice(itertools.combinations(avail, n), MAX_KICKER_COMBOS))


def _gen_all_raw(hand):
    cnt = hand_counts(hand)
    ranks = sorted(cnt)
    M = MoveType
    moves = []
    for r in ranks:
        c = cnt[r]
        moves.append(FMove(M.SINGLE, r, 1, (r,)))
        if c >= 2:
            moves.append(FMove(M.PAIR, r, 1, (r, r)))
        if c >= 3:
            moves.append(FMove(M.TRIO, r, 1, (r, r, r)))
        if c == 4:
            moves.append(FMove(M.BOMB, r, 1, (r, r, r, r)))
    if cnt.get(SMALL_JOKER) and cnt.get(BIG_JOKER):
        moves.append(FMove(M.ROCKET, BIG_JOKER, 1, (SMALL_JOKER, BIG_JOKER)))
    for r in ranks:
        if cnt[r] < 3:
            continue
        for k in ranks:
            if k == r:
                continue
            moves.append(FMove(M.TRIO_SINGLE, r, 1, (r, r, r, k)))
            if cnt[k] >= 2 and k <= 15:
                moves.append(FMove(M.TRIO_PAIR, r, 1, (r, r, r, k, k)))
    for win in _chain_windows(cnt, 1, 5, 12):
        moves.append(FMove(M.STRAIGHT, win[0], len(win), win))
    for win in _chain_windows(cnt, 2, 3, 10):
        moves.append(FMove(M.PAIR_CHAIN, win[0], len(win), win * 2))
    for win in _chain_windows(cnt, 3, 2, 6):
        t = len(win)
        body = win * 3
        moves.append(FMove(M.PLANE, win[0], t, body))
        rem = dict(cnt)
        for r in win:
            rem[r] -= 3
        banned = set(win)
        for kick in _kicker_singles(rem, banned, t):
            moves.append(FMove(M.PLANE_SINGLE, win[0], t, body + kick))
        for kick in _kicker_pairs(rem, banned, t):
            moves.append(FMove(M.PLANE_PAIR, win[0], t, body + tuple(kick) * 2))
    for r in ranks:
        if cnt[r] != 4:
            continue
        rem = dict(cnt)
        rem[r] = 0
        quad = (r, r, r, r)
        for kick in _kicker_singles(rem, {r}, 2):
            moves.append(FMove(M.QUAD_TWO_SINGLE, r, 1, quad + kick))
        for kick in _kicker_pairs(rem, {r}, 2):
            moves.append(FMove(M.QUAD_TWO_PAIR, r, 1, quad + tuple(kick) * 2))
    return moves


def _lead_order(m):
    # 自由出牌：先试张数多的（更快走完），同张数先小后大；炸弹放后面（留着控场）
    return (m.type in (T_BOMB, T_ROCKET), -m.n, m.main)


_GEN_CACHE = {}
_LEGAL_CACHE = {}
_CACHE_LIMIT = 400_000


def gen_all(hand):
    """手牌能打出的全部牌型（不含过），已按搜索用的启发式排好序。结果被缓存。"""
    res = _GEN_CACHE.get(hand)
    if res is None:
        if len(_GEN_CACHE) > _CACHE_LIMIT:
            _GEN_CACHE.clear()
        res = tuple(sorted(_gen_all_raw(hand), key=_lead_order))
        _GEN_CACHE[hand] = res
    return res


def beats_key(m, lk):
    """m 能否压过牌型键 lk=(type, main, length)。"""
    mt = m.type
    if mt == T_ROCKET:
        return True
    lt = lk[0]
    if lt == T_ROCKET:
        return False
    if mt == T_BOMB:
        return lt != T_BOMB or m.main > lk[1]
    if lt == T_BOMB:
        return False
    return mt == lt and m.length == lk[2] and m.main > lk[1]


def gen_beats(hand, lk):
    """能压过 lk 的所有着法（不含过），小的在前、炸弹最后。结果被缓存。"""
    ck = (hand, lk)
    res = _LEGAL_CACHE.get(ck)
    if res is None:
        if len(_LEGAL_CACHE) > _CACHE_LIMIT:
            _LEGAL_CACHE.clear()
        res = tuple(sorted((m for m in gen_all(hand) if beats_key(m, lk)),
                           key=lambda m: (m.type in (T_BOMB, T_ROCKET), m.main, m.n)))
        _LEGAL_CACHE[ck] = res
    return res


def gen_legal(hand, last):
    """last 为 None=自由出牌；否则为 FMove 或牌型键，返回 [过] + 能压的牌。"""
    if last is None:
        return list(gen_all(hand))
    lk = last.key if isinstance(last, FMove) else last
    return [PASS_MOVE] + list(gen_beats(hand, lk))


def move_from_text(text, hand=None):
    """把'334455'之类文字解析为 FMove（若多种解释取第一种；'过'/空 → PASS）。

    hand 给出时校验这手牌确实在手牌里。
    """
    t = str(text).strip()
    if t in ("", "过", "PASS", "pass", "不出", "要不起"):
        return PASS_MOVE
    cards = tuple(sorted(parse_cards(t)))
    h = hand_from_cards(cards)
    cands = [m for m in _gen_all_raw(h) if m.cards == cards]
    if not cands:
        raise ValueError(f"'{t}' 不是合法牌型")
    if hand is not None:
        for r, c in hand_counts(h).items():
            if count_of(hand, r) < c:
                raise ValueError(f"手牌里没有足够的 {RANK_NAMES[r]}")
    # 多解释时（如 4444+55 既是四带二单也是四带一对），优先四带二单等较常见者
    return cands[0]
