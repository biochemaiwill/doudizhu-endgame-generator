# -*- coding: utf-8 -*-
"""明牌残局求解 + 胜负手分析 + 胜率收敛曲线。

模块划分：
  fast.py     整数编码的快速牌型生成（规则与 doudizhu/moves.py 完全一致，改写为位运算以提速）
  solver.py   明牌残局求解器：AND/OR 搜索 + 置换表 + 着法排序 + 局面规范化
  tags.py     着法的"牌型操作"分类（拆对、拆顺、掉单张、留炸弹……）
  puzzles.py  预置教学残局
  scan.py     随机残局扫描：胜负手局面占比与常见操作
  winrate.py  整局对战 + 蒙特卡洛胜率曲线 + 失误分级
"""
from .fast import (FMove, PASS_MOVE, hand_from_cards, hand_from_text, hand_to_cards,
                   hand_to_text, hand_size, gen_all, gen_legal, move_from_text)
from .solver import Position, Solver, BudgetExceeded
