# 33IQ 507273 新一轮认证

完整三人公开原题：https://www.33iq.com/topic/574/7/ ，地主先出。
原题此前因独立参考求解超时而隔离。本轮在同一规则下增加单项时间预算，未修改供应商代码或参考规则。

结果：34个合法首着由优化求解器、关闭规范化的求解器、独立枚举参考求解器一致判定，仅单2必胜。
result.json 保存三套完整根动作签名；first-certification.json 保存首次成功结果。

建议试标4/4/4，理由和与5分的区分见annotation.json。目前证据支持反直觉首着和多分支后续，但不足以证明超出旧4分锚点的人类推理负担，故不标5分。未经真人校准。

关键分支：先2后若均过，则后续33个合法首着只有三个顺子可赢；若被大王接，经过牌权转移后，对方出7或9时，地主只能Q接，不能贪小用J。
after_joker_defense.json 保存该层全部10种领牌与地主胜/负应手。
示范线不是对手最优拖延线，强制过牌不算推理深度；完整胜负证据来自穷举根动作结果。

复现：
python3 research/expansion_v2_hard_retry/retry.py
python3 research/expansion_v2_hard_retry/inspect_branches.py
python3 research/expansion_v2_hard_retry/verify_artifacts.py

不训练模型、不修改任何旧标签或划分。来源、规则范围和局限沿用pilot100。
