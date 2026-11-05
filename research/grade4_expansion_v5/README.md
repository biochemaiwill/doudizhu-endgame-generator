# 高分变体试标

正式7道：6道4分、1道3分。全部与现有182题不重复，并从序列化输入重新三路逐根验证。保留原v0.1口径；分数未经真人校准。

6道4分只涉及3个既有父题家族（EX208、EX220、DZ098），不是6个独立新题族。全部必须与父题同分组，不允许跨train/val/test；本批未加入正在运行的scorer_v5。EX220地主8改6的版本保守降为3分。

正式数据dataset.jsonl，标签annotations.json，全部尝试和反例candidate_verification.json，关键回应key_branches.json，重新认证validation.json。verify_variants.py可重现候选验证，但会覆盖自己的候选文件。finalize.py重建正式数据。

本轮公开比赛来源检索未取得可核对的完整三家牌面，不声称提供职业赛事残局。
