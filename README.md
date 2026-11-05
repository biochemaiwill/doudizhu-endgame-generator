# 斗地主残局生成器

三人明牌必胜残局与人类推理难度评估研究。

当前阶段：数据调研。尚未实现生成器或训练评分模型。

- [数据调研报告](docs/data-research-2026-11-05.md)
- [来源版本元数据](docs/source-versions.json)

评分目标是人类解题难度与关键选择隐蔽程度。胜负、搜索成本和题目质量分别记录，不能混用。
代码、文档及小型清单优先存 GitHub；不在本地克隆大型数据、权重或第三方仓库。

- [六道三人残局试标与验证](research/pilot-v1/README.md)（等待评分口径确认）

- [100道三人残局试标：来源、评分与验证](research/pilot100/README.md)

- [牌面评分器v1：分层训练与结果](research/scorer_v1/README.md)（实验版，未达奖励模型要求）
- [新增高分候选](research/highscore_candidates_v1/README.md)（同族变体，未用于本轮训练）
