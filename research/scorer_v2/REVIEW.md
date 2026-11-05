# 验证与独立审查

- data tests：5通过；model tests：5通过；artifact tests：6通过。
- verify_deployment.py：17题端到端预测与封存结果最大绝对差2.2476291050566033e-7。
- 独立只读审查：family、相同URL、15pair无跨train/val/test；旧100仅train；TabPFN上下文仅train124。
- 独立重算test总分MAE0.4066884、宏MAE0.4813782、排除人工对照MAE0.5318233，匹配。
- 按验证集最小宏MAE选最终模型，选择记录先于test报告，未按测试反选。
- 保留EX220低估、人工对照规则贡献、旧8题在去对照子集更好等限制。
- 离线分享：153题、30个有效配对链接、JS语法检查通过；ZIP CRC、哈希、153数据、15pair通过。
- 新5分检索verify_candidates.py通过；本轮没有新5分。
- git diff --check — 通过。

改动范围：新增scorer_v2（数据合并、分组、特征与模型、训练、推理、结果、分享构建）、scorer_v2_prior_trial、grade5_search_v3，更新根入口。
不修改旧数据、旧模型或生产生成器。没有向外部推理服务发送数据。
