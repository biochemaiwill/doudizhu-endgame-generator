# 第二批验收与复核

已完成53条新标注：1分15、2分7、3分27、4分4、5分0。35条合成筛选后试标、2条新公开来源、1条此前未认证公开题、15条人工简单对照。
全部逐根动作完成优化、关闭规范化、独立枚举三路验证。新数据不与旧100、6锚点或上轮候选规范化重复。
15对照均保持三家各自牌数、角色、行动人、自由首出状态一致，双方共享family。对照不保持原始牌点集合且人为一手可出完；只用于基础诊断。

独立只读复核：全部记录的合法/获胜动作、计数及证据一致；公开来源3题的关键分支有额外核对。
修正EX220“队友需拆J或Q”为“队友需出J或Q”（队友本来是单张）。标签档位和胜负结果未改变。

验证命令：
- PYTHONDONTWRITEBYTECODE=1 python3 research/expansion_v2/tests.py — 6通过。
- python3 research/expansion_v2_hard_retry/verify_artifacts.py — 通过：三套34动作签名完全一致、仅2胜、关键7/9防守只Q胜、4/4/4。
- git diff --check — 通过。
- 构建时assemble.py从最终输入重新完成53条三路验证；逐条耗时保存在evidence.json。

改动仅新增expansion_v2、expansion_v2_sources、expansion_v2_hard_retry和根README入口；旧数据、训练配置、模型不变。
环境、种子、数据哈希见manifest.json；完整搜索日志无损压缩保存，压缩前哈希可核对。
没有5分证据，没有人类难度校准，不能宣称新增题都来自自然对局或代表独立的真实玩家分布。
