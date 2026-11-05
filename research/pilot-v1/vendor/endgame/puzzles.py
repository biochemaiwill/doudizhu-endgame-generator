# -*- coding: utf-8 -*-
"""预置教学残局。

每题的答案都由求解器给出、并经 tests/reference_solver.py（独立的暴力搜索）复核；
讲解文字是人工对照求解器的主变例写的。牌面记号：T=10，X=小王，D=大王。

字段：
  hands   各家手牌文字（0 号是地主；二人版是"我方 vs 对手"）
  turn    轮到谁出（0 地主 / 1 农民甲=地主下家 / 2 农民乙=地主上家）
  last    需要压的牌（None = 自由出牌）；passes 为已连续过牌家数
  answer  唯一必胜着法（文字）

前 8 题是教学残局，第 9 题是自造题（用本项目的求解器从自对弈残局里挖出来的“唯一胜着”局面）。
"""
from dataclasses import dataclass

from .fast import hand_from_text, move_from_text
from .solver import Position


@dataclass
class Puzzle:
    key: str
    title: str
    hands: tuple
    turn: int
    answer: str
    explain: str
    last: str = None
    passes: int = 0
    source: str = "本项目构造 / 求解器挖掘"

    def position(self):
        hs = tuple(hand_from_text(h) for h in self.hands)
        last = move_from_text(self.last) if self.last else None
        return Position(hs, self.turn, last, self.passes)


PUZZLES = [
    Puzzle("p1_drop_single", "二人 · 先掉小单张",
           ("2 K K 3", "A A 4"), 0, "3",
           "我方 KK 怕对手的 AA，3 怕对手的 4。先把最小的 3 送出去：对手用 4 管，我用 2 夺回出牌权再出 KK；"
           "对手拆 AA 用 A 管，他就没有对子了，2 夺回后 KK 走完。先出 KK 会被 AA 管住，对手随后打出最后一张 4；"
           "先出 2 则之后 3 和 KK 总有一手被对手抓住。"),
    Puzzle("p2_split_pair", "二人 · 拆对子引出对手的 2",
           ("T T K K D", "4 4 4 J 2 2"), 0, "K",
           "对手能管对子的只有 22，而他的 444+J 一手就能走完，所以绝不能把出牌权交给他。拆开 KK 出单 K："
           "对手不管，我再出 K、大王，最后 TT 一手走完；对手拆 22 用单 2 管，他再也凑不出对子，大王夺回后 TT、K 安全。"
           "直接出 TT 或 KK 都会被 22 管住并让对手走完。"),
    Puzzle("p3_trio_to_chain", "二人 · 拆三张组连对",
           ("3 6 6 7 7 8 8 8", "5 7 A A A"), 0, "6 6 7 7 8 8",
           "对手的 AAA 能管住我任何三张、三带一、三带二，而且带上 5 或 7 就走完。把 888 拆开，与 66、77 组成连对，"
           "对手管不住；剩下 3 和 8 两张单牌，先出 3，对手无论用什么管，最后都拦不住我的 8。"),
    Puzzle("p4_control_with_two", "二人 · 先出 2 保住出牌权",
           ("K A A A 2", "T T 2 2 2"), 0, "2",
           "先出 AAA+K 会被对手 222+10 管住，他剩一张 10 直接走完。单张 2 对手压不住（同点数不能压），"
           "出牌权留在我手里，剩下的 AAA+K 一手走完。"),
    Puzzle("p5_kicker_choice", "二人 · 带对子不带单张",
           ("3 3 4 4 4 8", "9 T T T A"), 0, "3 3 4 4 4",
           "对手的 TTT 能管三带一，却管不了三带二（他没有对子）。444 带 33 打出，只剩一张 8 直接走完；"
           "444 带 8 则对手 TTT 带一张管住，再出最后一张单牌获胜。"),
    Puzzle("p6_feed_teammate", "三人 · 农民拆三张给队友送对",
           ("7 7 7 2", "3 3 3 6 T", "Q Q"), 1, "3 3",
           "轮到农民甲出牌，下一个出牌的是只剩一对 QQ 的农民乙。拆开 333 打出一对 3，队友用 QQ 接手直接走完。"
           "出 333 带牌会被地主 777+2 管住走完；出单张队友接不上。团队博弈里，最好的一手常常是让队友赢。"),
    Puzzle("p7_farmer_keep_king", "三人 · 农民先掉 4、留 K 看住地主",
           ("9 9", "8 8 8", "3 3 4 K"), 2, "4",
           "地主只剩 99。农民乙若先出 33，地主用 99 直接走完；先出 K，之后 4 被地主拆 9 管住，地主剩最后一张 9 走完。"
           "先出 4：地主不管，就接着出 K、再出 33 走完；地主拆 99 用 9 管，K 压回，地主那张 9 再也管不住 33。"),
    Puzzle("p8_small_pair_first", "三人 · 地主先出小对，AA 回收",
           ("Q Q 5 A A", "6 2", "8 K K"), 0, "Q Q",
           "地主有 QQ、AA、5 三手牌，5 一出就会被 6、2、8 管，所以必须最后出。先出 QQ：农民乙用 KK 管，地主 AA 压回、"
           "再出 5 走完；没人管，就 AA、5 连走。先出 AA 则 QQ 会被 KK 管住，农民乙再出 8 走完。"),
    Puzzle("p9_force_split", "三人 · 自造题 · 先出单 Q，逼地主拆对 A",
           ("J A A D", "6 T Q K", "T Q 2 2"), 2, "Q",
           "地主只剩 J、A、A、大王，农民这边能管住 A 的只有农民乙的两张 2。农民乙先出单 Q：地主的 J 管不住，"
           "只好拆开 AA 用一张 A 管；农民乙用 2 压回，地主再用大王夺回、出 J，农民甲用 Q 管住，农民乙最后一张 2 压住，"
           "再出 10 走完。先出 10：地主用 J 管，农民甲 Q、农民乙 2 依次压上，地主大王压回后一手对 A 走完；"
           "先出对 2 虽然没人要得起，却把两张能管 A 的牌一次用掉，地主的 A、A、大王从此没人能管。",
           source="本项目求解器挖掘（自造题，答案经独立暴力搜索复核）"),
]

ALL_PRESETS = list(PUZZLES)
BY_KEY = {p.key: p for p in ALL_PRESETS}
