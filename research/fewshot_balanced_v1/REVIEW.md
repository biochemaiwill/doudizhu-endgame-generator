# 验证与独立复核

- 执行：/home/wutianxiang/miniconda3/envs/research/bin/python research/fewshot_balanced_v1/tests.py — 7通过。
- 执行：/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v1/tests.py — 9通过。
- 执行：python3 research/pilot100/tests.py — 11通过。
- 执行：/home/wutianxiang/miniconda3/envs/research/bin/python research/fewshot_balanced_v1/predict.py research/scorer_v1/example.json — 通过，输出有限1—4分，5分unsupported、reward false。
- git diff --check — 通过。
- 独立只读审查：每次支持集8道、各级2道、均属于原训练集；验证/测试互斥，标准化仅用支持集。
- 独立重算50个导出模型共100组验证/测试指标，MAE和宏MAE在1e-6内匹配。
- 同族高分支持、单道高分测试、开发集曝光及对照混杂均已披露，无阻塞问题。
- 按复核建议收窄METHODS结论，不由单次实验推断任何8样本方案都不可能有效。

改动范围：新增fewshot_balanced_v1的实验、推理、测试、导出产物、环境、结果及方法报告，更新根README入口；未改动原冻结数据与结果。
所有新增实验使用CPU，未安装依赖或下载模型。尚未验证普通玩家难度、5分、生成器奖励有效性。
