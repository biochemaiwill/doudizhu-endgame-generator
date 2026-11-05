# CE 四分类开发选型 v6

本目录只使用既有189条开发语料，不读取或评估strict_holdout_v6。
所有神经网络均以四类交叉熵训练，输出[batch,3,4] logits；推理先softmax，再argmax+1。不再采用期望分数取整。
总体分类是主任务，推理深度和隐蔽性两个分类头各以0.25权重辅助训练。
树分类器使用log_loss分裂准则，各头独立拟合。

## 分组与选型
数据由scorer_v4的182条及grade4_expansion_v5的7条合并。
同family、规范化后相同来源URL、24组配对和显式parent_id取连通分量，不能跨fold。
MasterMind大组始终仅训练；其他样本3折×2随机种子，6个不同验证集合，均含4档。
因此CV结果针对非MasterMind开发题族，不是全来源随机总体估计；开发语料曾被研究，不是新独立最终测试。

20个固定候选：原143维及191维关系特征的ExtraTrees/RandomForest、自然/均衡、leaf1/3；
另有MLP与小型cross attention的自然/均衡CE配置。
关系特征只根据三家已知手牌计算不同牌型数量与相对控制点数，不访问标签、文字依据或求解器结果。
NN固定100epochs、lr0.001、AdamW、剪裁梯度5；未对外层fold早停。CV均值accuracy主选，macro recall打破平局。
cross/MLP汇总特征标准化仅拟合对应fold train；cross模型前45维保留原始计数以免破坏count embedding。

## 产物与运行
- train.py：冻结小网格CV、内部选择及用189开发题最终拟合。
- model.py：输入特征、CE网络、指标及推理。
- predict.py：读取一道或一组JSON牌面，返回四类概率和argmax等级。
- cv_split.json：完整连通组和每fold ID。
- cv_results.json：所有开发候选的accuracy/recall/confusion。
- selection.json：只表示本目录候选内部选择，外部分类器可由主线程统一比较。
- model.joblib.gz：内部选中模型；仅加载本项目生成的可信文件。
- environment.json：版本、随机种子、数据hash、耗时。

命令：
/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v6/train.py
/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v6/tests.py
/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v6/predict.py input.json

训练拒绝覆盖selection.json。评估90%目标只能由新封存测试集决定，不能使用训练准确率或开发CV冒充。
没有5分训练样本，只有1—4四分类；所有人工评分仍为助手弱标签，尚未真人校准。

## 已完成的内部候选结果

| 配置 | CV平均准确率 | CV平均宏召回 |
|---|---:|---:|
| kind-forest_mode-base_balanced-True_depth-None_leaf-1_max_features-sqrt | 57.5316% | 46.7720% |
| kind-forest_mode-base_balanced-False_depth-None_leaf-3_max_features-sqrt | 57.1924% | 46.4040% |
| kind-forest_mode-base_balanced-False_depth-None_leaf-1_max_features-sqrt | 56.4601% | 46.0239% |
| kind-forest_mode-base_balanced-True_depth-None_leaf-3_max_features-sqrt | 56.1363% | 45.4534% |
| kind-forest_mode-relational_balanced-True_depth-None_leaf-1_max_features-sqrt | 55.3962% | 45.0380% |
| kind-forest_mode-relational_balanced-True_depth-None_leaf-3_max_features-sqrt | 54.6870% | 44.7955% |

内部选择：kind-forest_mode-base_balanced-True_depth-None_leaf-1_max_features-sqrt。未达到90%目标；其六折4分召回均为0。CLI原示例仅为序列化/推理烟雾检查，不是测试集结果。外部TabPFN分类器与新封存测试由主线程统一比较，不在本目录宣称严格验收通过。
