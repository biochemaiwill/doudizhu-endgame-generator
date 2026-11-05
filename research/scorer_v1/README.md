# 牌面评分器 v1：冻结100题先导实验

## 结论
训练流程完成，但模型尚不能承担高难度筛选或生成器奖励。验证集选择了只使用三家剩余牌数的岭回归；牌面岭回归和小型神经网络未胜过它。测试排名相关系数不能单独代表成功。

## 数据和划分
沿用100题单助手弱标注，原数据不变。输出overall、reasoning_depth、move_concealment。1—4分有样本，5分无样本，不支持5分。

|集合|1分|2分|3分|4分|总数|
|---|---:|---:|---:|---:|---:|
|train|30|27|11|2|70|
|val|6|6|2|1|15|
|test|6|6|2|1|15|

50道缺少对局编号的MasterMind全部留train；同一Gamersky文章家族全部留train。其余按总分分层。val/test是小样本跨来源开发评估，不是独立真人测试或同分布随机样本。只有一题4分在val、一题4分在test，无法估计稳定的高分召回率。

## 模型和选择
输入为三家15牌点计数及轮次、桌面牌型/主点/链长/牌张、连续过牌数等，共86维。手牌内部排序不影响编码，三个座位不交换。特征不包含来源、解释、标签、获胜出法或求解结果。推理只检查牌面，不运行求解器。

比较训练均值、仅三家张数的岭回归、全牌面岭回归、小型三头序数MLP。MLP为86→32→16→3加有序阈值；固定种子17/41/73，CPU单线程；最多300轮、验证早停40轮、ordinal BCE。岭回归alpha=1/10/100。归一化只fit train。先按val总分MAE选模（平局比三头平均MAE），写入选择记录后才评估test。

|模型|val总分MAE|test总分MAE|test Spearman|test等级准确率|
|---|---:|---:|---:|---:|
|train_mean|0.710|0.710|—|40.0%|
|ridge_counts_1|0.582|0.540|0.774|60.0%|
|ridge_board_100|0.699|0.738|-0.052|40.0%|
|ordinal_mlp_ensemble|0.633|0.718|0.029|53.3%|

全部候选指标见runs/frozen100_20261105/metrics.json。所选模型ridge_counts_1。测试只评估本轮冻结候选，没有根据test重新调参。

## 高分失败
所选模型把test两个3分题均判为2分档，唯一4分题判为3分档；DZ095连续预测约2.724，弱标签是4。整体MAE0.540掩盖不了高分低估。model.json和predict.py明确给出ready_for_generation_reward=false。
这些结果不能推出“多用GPU或更大网络就会改善”。需先补高分样本、审查标签、明确家族划分。此test已被查看，后续反复用它改模型只能算开发实验。

## 使用
在仓库根目录：
    /home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v1/predict.py research/scorer_v1/example.json

example.json是DZ095，用来展示低估。输出三个连续分数和以下状态：supported_grades=[1,2,3,4]，grade_5=unsupported_no_training_examples，human_calibrated=false，ready_for_generation_reward=false。
输入合法不代表目标阵营必胜，生成筛选仍须独立验证可解性。

## 验证和复现
    /home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_v1/tests.py

9项测试覆盖牌序不变性、座位区别、标签不进特征、非法牌张/牌型、分层和家族隔离、序数一致性、100题参数重载一致及归一化train-only。原pilot100有11项测试。
train.py保存划分、训练/验证曲线、选模记录、全部候选指标、逐题预测和小型JSON参数，不保存逐轮权重。
训练命令为同一Python环境执行research/scorer_v1/train.py。已有run目录时拒绝覆盖；重跑须使用独立研究副本或明确的新运行目录，不删除本轮记录。软件版本见metrics.json。

## 范围
没有GPU、新依赖安装、付费API调用或本地电脑项目文件新增。不修改用户另一台电脑上的生成器。并行高分候选在research/highscore_candidates_v1；没有用于本轮训练/验证/测试。
