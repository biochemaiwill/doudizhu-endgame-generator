# 182题有序四分类：注意力架构试验

用户批准按验证准确率选模，保留4档；没有5分样本，因此不做五分类。

## 实验协议

- 182题按family、相同来源URL及24组配对取连通分量；MasterMind大组留train。
- train124（39/39/42/4）；val29、test29均9/9/9/2。
- 旧语料与测试曾被研究过，本次为重分组开发对照，不是独立最终测试，不能与旧17题指标直接横比。
- MLP、45-token全局自注意力、每手牌自注意力后双向跨手牌注意力，均自然/逆频率权重两种。
- attention维度64、4头、2层；无预训练、无一手结束规则。
- 每种配置3种子17/41/73平均；val accuracy优先，macro recall和MAE打破平局；封存选择后才算test。
- 固定解码：有序期望分数加0.5取整，裁剪至1—4。

## 结果

| 模型 | 验证准确率 | 测试准确率 | 测试宏召回 | 测试MAE |
|---|---:|---:|---:|---:|
| mlp_natural | 44.8% | 44.8% | 36.1% | 0.595 |
| mlp_balanced | 62.1% | 72.4% | 58.3% | 0.470 |
| self_natural | 51.7% | 51.7% | 41.7% | 0.577 |
| self_balanced | 48.3% | 41.4% | 33.3% | 0.691 |
| cross_natural | 51.7% | 55.2% | 44.4% | 0.561 |
| cross_balanced | 44.8% | 55.2% | 44.4% | 0.589 |

验证集选择 **mlp_balanced**。测试准确率72.4%，宏召回58.3%。

各类别召回：1分88.9%、2分77.8%、3分66.7%、4分0%。两道4分均被判为2分，模型尚不适合作高难题奖励。

混淆矩阵（行是真实1—4分、列是预测1—4分）：

[[8, 1, 0, 0], [0, 7, 2, 0], [0, 3, 6, 0], [0, 2, 0, 0]]

注意力配置此次未胜过MLP，不能宣称架构升级有效。小样本、有序解码、训练4分仅4例都可能影响结果，不能仅归因于attention无用。尚未实施保胜动作辅助预训练或新配对损失。

## 复现与检查

训练：/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v4/train.py
测试：同Python research/scorer_v4/tests.py（3项通过，覆盖所有分组边、排列不变、序数单调、选择规则和指标复算）。
导出：model.json.gz；训练时重新加载3个selected state_dict预测与原预测一致（1e-6）。
推理：同Python research/scorer_v4/predict.py input.json（已用原example.json smoke通过）。
results.json保存全部候选结果、种子、epoch、环境及耗时；split.json保存完整组成员和分配。
脚本拒绝覆盖已有results.json，复跑用隔离输出目录。
