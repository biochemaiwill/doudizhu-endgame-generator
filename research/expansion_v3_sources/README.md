# 第三批公开源后继候选
按原v0.1保守选4条：3条2分、1条3分，无4/5分。selection.json含hands/turn/last/passes/family/label及完整根动作证据；不自动进入训练。
两个公开原始牌均为20/17/17：百度2013最后牌例、小米2022-08-26。原始全局胜负未知；从给定牌面生成合法非终局前缀并单独认证，不宣称原局地主必胜。同源所有后继必须同划分。
1000次尝试得3个三路验证候选；另交换PUB3-0907中地主J和下家大王得到2分对照。交换后的状态不是原文真实后继，明确标synthetic_card_swap_variant。
该对为3分vs2分，三家牌数均[7,12,4]且总牌点集合、turn/last/passes完全相同；双方都不是一手出完题。
每条完整根动作的优化、关闭canonical及独立枚举签名相等，finalize.py再验证前缀/换牌守恒、去重与非一手终局。标签未经真人校准，没有按搜索量抬成5分。
复现：python3 research/expansion_v3_sources/mine.py；对照的生成记录保留swap-candidates.json；python3 research/expansion_v3_sources/finalize.py。
筛选日志无损压缩，source-search-log.json记录查询及排除理由。旧数据及训练未修改。
