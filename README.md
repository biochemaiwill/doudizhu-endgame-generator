# 斗地主残局生成器

三人明牌必胜残局与人类推理难度评估研究。

当前阶段：研究暂停，334题及全部对照结果已整理，90%目标未达。

接手请先读 [HANDOFF.md](HANDOFF.md)，完整产物见 [最终Release](https://github.com/biochemaiwill/doudizhu-endgame-generator/releases/tag/final-handoff-20261105)。

- [数据调研报告](docs/data-research-2026-11-05.md)
- [来源版本元数据](docs/source-versions.json)

评分目标是人类解题难度与关键选择隐蔽程度。胜负、搜索成本和题目质量分别记录，不能混用。
代码、文档及小型清单优先存 GitHub；不在本地克隆大型数据、权重或第三方仓库。

- [六道三人残局试标与验证](research/pilot-v1/README.md)（等待评分口径确认）

- [100道三人残局试标：来源、评分与验证](research/pilot100/README.md)

- [牌面评分器v1：分层训练与结果](research/scorer_v1/README.md)（实验版，未达奖励模型要求）
- [新增高分候选](research/highscore_candidates_v1/README.md)（同族变体，未用于本轮训练）

- [均衡few-shot实验与方法调研](research/fewshot_balanced_v1/README.md)（每级2题，5次抽样；仍无5分样本）

- [第二批扩充：53道与15组等牌数对照](research/expansion_v2/README.md)（新增31道3—4分，仍无5分；未加入旧训练）

- [合并153题训练与模型比较](research/scorer_v2/README.md)（题族/配对隔离；含离线分享构建脚本）

- [第三批29题、严格反事实配对与注意力架构调研](research/expansion_v3/README.md)（总计182题，旧模型冻结）

- [CE四分类与90%目标验收状态](research/scorer_v6/FINAL_REPORT.md)（当前未达标，不以开发CV冒充新测试）

- [机制型配对样板：四机制、完整正反例](research/mechanism_pairs_v1/README.md)（8题中仅1题新增，未加入训练）

- [334题扩充与固定模型对照](research/data300_v1/README.md)（本批未带来准确率提升，保留完整结果）
