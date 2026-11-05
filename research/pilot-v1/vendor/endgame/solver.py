# -*- coding: utf-8 -*-
"""明牌残局求解器（完全信息，二人 / 三人斗地主）。

局面 = (各家手牌, 轮到谁, 需要压的牌型键, 已连续过牌数)
  - 0 号永远是地主（二人版里是"我方"），1、2 号是农民（同队）。出牌顺序 0→1→2→0。
  - 需要压的牌只保留 (牌型, 主点数, 链长) 三元组：带的是什么翅膀不影响后续。
  - 连续 n-1 家过牌后，最后出牌的人重新自由出牌。

算法：AND/OR 树搜索（负极大值在"只有胜/负两值、且有两支队伍"时的写法）
  - 地主结点：任一着法能让地主赢 → 地主赢（OR），找到一个就剪枝；
  - 农民结点：任一着法能让农民赢 → 农民赢（AND 的对偶），找到一个就剪枝。
  这就是 alpha-beta 在 {0,1} 值域上的特例：窗口永远是 (0,1)，第一个"够好"的着法立即截断。
  - 置换表：规范化局面 → 地主胜负（bool）。不同出牌顺序到达同一局面只算一次。
  - 着法排序（默认 order="plain"）：能一手出完的直接判胜；自由出牌时先试张数多的、同张数先小后大，
    炸弹放最后；压牌时先试刚好压住的小牌，炸弹再次，"过"最后。
    另外实测过两种"聪明"排序：对手压不住的牌优先（"smart"）、接队友的牌先过（"tm"），
    在随机残局上结点数的中位数略少，但最坏情况反而多出数倍，所以默认不用（仍可选）。
  - 局面规范化（点数压缩）：场上已无人持有的点数被删去，只在 3..A 区间保留一个"断点"
    防止顺子被错误接上；2 和大小王保持原值。于是"打掉 5 留 6"与"打掉 6 留 5"这类
    同构局面共用一条置换表记录。需要压的主点数若已无人持有，映射成相邻点数 +0.5，
    保持大小关系不变。

对外接口：
  Solver().landlord_wins(pos)           → bool
  Solver().move_values(pos)             → [(着法, 行动方是否必胜), ...]   —— 胜负手分析的核心
  Solver().principal_line(pos)          → 双方最佳着法序列（示范线）
"""
import sys
import time
from dataclasses import dataclass, field

from .fast import (FMove, PASS_MOVE, gen_all, gen_beats, hand_size, hand_to_text,
                   T_BOMB, T_ROCKET, _SHIFT)

sys.setrecursionlimit(10000)

ROLE_NAMES_3 = ("地主", "农民甲(地主下家)", "农民乙(地主上家)")
ROLE_NAMES_2 = ("地主(我方)", "农民(对手)")


class BudgetExceeded(Exception):
    """搜索结点数超过预算（供蒙特卡洛调用方降级为快速推演）。"""


@dataclass
class Position:
    hands: tuple                 # 各家手牌（int 编码），0 号是地主
    turn: int = 0                # 轮到谁
    last: FMove = None           # 需要压的牌；None = 自由出牌
    passes: int = 0              # 自 last 打出后已连续过牌的家数
    names: tuple = field(default=None)

    def __post_init__(self):
        self.hands = tuple(self.hands)
        if self.names is None:
            self.names = ROLE_NAMES_3 if len(self.hands) == 3 else ROLE_NAMES_2
        if self.last is not None and self.last.is_pass:
            self.last = None

    @property
    def n(self):
        return len(self.hands)

    @property
    def last_mover(self):
        if self.last is None:
            return None
        return (self.turn - 1 - self.passes) % self.n

    def side(self, player):
        return "landlord" if player == 0 else "farmer"

    def play(self, move):
        """返回走完 move 后的新局面（不检查合法性）。"""
        n = self.n
        nt = (self.turn + 1) % n
        if move.is_pass:
            if self.passes + 1 >= n - 1:
                return Position(self.hands, nt, None, 0, self.names)
            return Position(self.hands, nt, self.last, self.passes + 1, self.names)
        hands = list(self.hands)
        hands[self.turn] -= move.delta
        return Position(tuple(hands), nt, move, 0, self.names)

    def legal(self):
        hand = self.hands[self.turn]
        if self.last is None:
            return list(gen_all(hand))
        return [PASS_MOVE] + list(gen_beats(hand, self.last.key))

    def winner_after(self, move):
        """若 move 直接出完手牌，返回获胜方 'landlord'/'farmer'，否则 None。"""
        if not move.is_pass and self.hands[self.turn] == move.delta:
            return self.side(self.turn)
        return None

    def describe(self):
        lines = []
        for i, h in enumerate(self.hands):
            mark = " ←轮到" if i == self.turn else ""
            lines.append(f"{self.names[i]}：{hand_to_text(h)}（{hand_size(h)} 张）{mark}")
        if self.last is None:
            lines.append("需要压的牌：无（自由出牌）")
        else:
            lines.append(f"需要压的牌：{self.last.label()}（{self.names[self.last_mover]} 打出）")
        return "\n".join(lines)


# ---------------------------------------------------------------- 规范化
_NIB_LOW = int("1" * 15, 16)                       # 每个 4bit 组的最低位
_CHAIN_MASK = sum(1 << _SHIFT[r] for r in range(3, 15))   # 3..A 的最低位


def _nz(x):
    """每个 4bit 组非零 → 该组最低位为 1。"""
    x = x | (x >> 1)
    x = x | (x >> 2)
    return x & _NIB_LOW


_HIGH_MASK = 0xFFF << _SHIFT[15]                  # 2、小王、大王三组：原样保留
_CANON_CACHE = {}


def _canon_plan(pm):
    """按"哪些 3..A 点数还有人持有"(pm) 预计算压缩方案，结果缓存。"""
    mapping = {}
    nxt = 3
    gap = False
    for r in range(3, 15):
        if (pm >> _SHIFT[r]) & 1:
            if gap and nxt > 3:
                nxt += 1
            mapping[r] = nxt
            nxt += 1
            gap = False
        else:
            gap = True
    identity = all(mapping[r] == r for r in mapping)
    # 连续持有的一段点数整体平移：(源位移, 段掩码, 目标位移)
    runs = []
    r = 3
    while r < 15:
        if r in mapping:
            s = r
            while r + 1 in mapping and r + 1 < 15:
                r += 1
            width = r - s + 1
            runs.append((_SHIFT[s], (1 << (4 * width)) - 1, _SHIFT[mapping[s]]))
        r += 1
    mainmap = {}
    for x in range(3, 15):
        if x in mapping:
            mainmap[x] = mapping[x]
        else:
            below = [mapping[y] for y in mapping if y < x]
            mainmap[x] = (max(below) if below else 2) + 0.5
    return identity, tuple(runs), mainmap


def _canon(hands, lk):
    """点数压缩。返回 (新 hands, 新 lk)。见模块文档。"""
    total = 0
    for h in hands:
        total |= h
    pm = _nz(total) & _CHAIN_MASK
    plan = _CANON_CACHE.get(pm)
    if plan is None:
        plan = _canon_plan(pm)
        _CANON_CACHE[pm] = plan
    identity, runs, mainmap = plan
    if lk is not None and lk[1] < 15:
        lk = (lk[0], mainmap[lk[1]], lk[2])
    if identity:
        return hands, lk
    new_hands = []
    for h in hands:
        nh = h & _HIGH_MASK
        for ss, mask, ds in runs:
            nh |= ((h >> ss) & mask) << ds
        new_hands.append(nh)
    return tuple(new_hands), lk


class Solver:
    """带置换表的明牌求解器。一个实例的置换表可在多次求解间复用。"""

    def __init__(self, node_limit=None, canonical=True, tt_limit=4_000_000, order="plain"):
        self.tt = {}
        self.nodes = 0
        self.node_limit = node_limit
        self.canonical = canonical
        self.tt_limit = tt_limit
        self.order = order
        self.deadline = None     # time.perf_counter() 截止时刻；超时同样抛 BudgetExceeded

    # ------------------------------------------------------------ 内核
    def _search(self, hands, turn, lk, passes):
        # 置换表键：把三家手牌和 (轮次, 牌型键, 过牌数) 打包成一个大整数（比嵌套 tuple 省内存）
        key = 0
        for h in hands:
            key = (key << 60) | h
        if lk is None:
            key = (key << 20) | (turn << 1) | passes
        else:
            key = (key << 20) | (((lk[0] << 11) | (int(lk[1] * 2) << 4) | lk[2]) << 4) \
                | 8 | (turn << 1) | passes
        v = self.tt.get(key)
        if v is not None:
            return v
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise BudgetExceeded()
        if self.deadline is not None and not (self.nodes & 0x3FF) and time.perf_counter() > self.deadline:
            raise BudgetExceeded()          # 每 1024 个结点看一次钟（复盘用：机器忙时结点速度会掉一半）

        n = len(hands)
        hand = hands[turn]
        is_ll = turn == 0
        nt = turn + 1
        if nt == n:
            nt = 0

        if lk is None:
            moves = gen_all(hand)
        else:
            moves = gen_beats(hand, lk)
        # 1) 一手出完 → 直接赢
        for m in moves:
            if m.delta == hand:
                self._store(key, is_ll)
                return is_ll

        # 2) 着法排序
        order = self.order
        if order == "smart":
            moves = self._ordered(hands, turn, lk, passes, moves)
        elif lk is not None:
            if order == "pass_first":
                moves = (PASS_MOVE,) + moves
            elif order == "tm" and turn != 0 and (turn - 1 - passes) % n != 0:
                moves = (PASS_MOVE,) + moves
            else:
                moves = moves + (PASS_MOVE,)

        result = not is_ll
        canon = self.canonical
        if canon:
            others = 0
            for i, h in enumerate(hands):
                if i != turn:
                    others |= h
            others_nz = _nz(others)
        for m in moves:
            if m.type == 0:  # 过
                if passes + 1 >= n - 1:
                    child = self._search(hands, nt, None, 0)
                else:
                    child = self._search(hands, nt, lk, passes + 1)
            else:
                nh = list(hands)
                left = hand - m.delta
                nh[turn] = left
                nh = tuple(nh)
                mk = m.key
                # 只有当某个 3..A 的点数被打光（全场无人再持有）时才需要重新压缩
                if canon and (_nz(m.delta) & ~others_nz & ~_nz(left) & _CHAIN_MASK):
                    nh, mk = _canon(nh, mk)
                child = self._search(nh, nt, mk, 0)
            if child == is_ll:
                result = is_ll
                break
        self._store(key, result)
        return result

    def _store(self, key, val):
        if len(self.tt) >= self.tt_limit:
            self.tt.clear()
        self.tt[key] = val

    def _ordered(self, hands, turn, lk, passes, moves):
        n = len(hands)
        if turn == 0:
            opps = hands[1:]
        else:
            opps = (hands[0],)
        if lk is not None:
            last_mover = (turn - 1 - passes) % n
            if turn != 0 and last_mover != 0:
                # 队友出的牌：先试"不压"
                return (PASS_MOVE,) + moves
        # 对手压不住的牌优先（保住出牌权）；其余按生成顺序（自由出牌：张数多的先；压牌：小的先）
        safe = []
        rest = []
        for m in moves:
            mk = m.key
            for oh in opps:
                if gen_beats(oh, mk):
                    rest.append(m)
                    break
            else:
                safe.append(m)
        if lk is None:
            return safe + rest
        safe.extend(rest)
        safe.append(PASS_MOVE)
        return safe

    def _enter(self, pos):
        hands, lk = pos.hands, (pos.last.key if pos.last is not None else None)
        if self.canonical:
            hands, lk = _canon(hands, lk)
        return hands, lk

    # ------------------------------------------------------------ 公共接口
    def landlord_wins(self, pos):
        hands, lk = self._enter(pos)
        return self._search(hands, pos.turn, lk, pos.passes)

    def value_after(self, pos, move):
        """走 move 后地主是否必胜。"""
        w = pos.winner_after(move)
        if w is not None:
            return w == "landlord"
        return self.landlord_wins(pos.play(move))

    def move_values(self, pos):
        """每个合法着法 → 行动方（所在队伍）是否必胜。不剪枝，逐个求解。"""
        out = []
        is_ll = pos.turn == 0
        for m in pos.legal():
            ll = self.value_after(pos, m)
            out.append((m, ll == is_ll))
        return out

    def best_move(self, pos):
        """返回 (着法, 行动方是否必胜)。必胜时给一个必胜着法；必败时给'最顽强'的着法（出牌最多者）。"""
        is_ll = pos.turn == 0
        legal = pos.legal()
        for m in legal:
            if pos.winner_after(m):
                return m, True
        for m in self._ordered_positions(pos, legal):
            if self.value_after(pos, m) == is_ll:
                return m, True
        # 必败：挑一个"看起来最像样"的着法
        non_pass = [m for m in legal if not m.is_pass]
        if non_pass:
            return max(non_pass, key=lambda m: (not m.is_bomb, m.n, -m.main)), False
        return PASS_MOVE, False

    def _ordered_positions(self, pos, legal):
        """与内核同样的排序，但作用在原始（未规范化）着法上。"""
        n = pos.n
        is_ll = pos.turn == 0
        opps = [p for p in range(n) if (p == 0) != is_ll]
        if pos.last is not None:
            lm = pos.last_mover
            if lm != 0 and not is_ll:
                return legal                        # 过 已在首位
        safe = [m for m in legal if not m.is_pass and
                all(not gen_beats(pos.hands[p], m.key) for p in opps)]
        rest = [m for m in legal if not m.is_pass and m not in safe]
        tail = [PASS_MOVE] if pos.last is not None else []
        return safe + rest + tail

    def principal_line(self, pos, max_plies=80):
        """双方都按求解器最佳着法走到底，返回 [(玩家, 着法, 走前地主是否必胜), ...]。"""
        line = []
        cur = pos
        for _ in range(max_plies):
            ll_before = self.landlord_wins(cur)
            m, _ = self.best_move(cur)
            line.append((cur.turn, m, ll_before))
            if cur.winner_after(m):
                break
            cur = cur.play(m)
        return line


def solve_position(pos, **kw):
    """便捷函数：返回 (地主是否必胜, 求解结点数)。"""
    s = Solver(**kw)
    v = s.landlord_wins(pos)
    return v, s.nodes
