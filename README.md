# 斗地主残局生成器

三人明牌必胜残局与人类推理难度评估研究。

当前阶段：完成100题试标、评分器基线及1—4分均衡few-shot开发实验；尚未训练生成器。

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
