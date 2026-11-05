# 项目交接（2026-11-05）

研究暂停。原本机和服务器项目副本将在完整归档和恢复验证后删除。请从本仓库及最终Release接手。

## 结果与限制

- 最新主数据 research/data300_v1/dataset_merged.jsonl：334题，1/2/3/4分分别76/175/69/14。SHA256 c8d107007c9ea6ca3a05416a31b4f56b92996820aa22ef7699377fd75ac133e0。

- 没有可靠5分题。难度是助手试标，尚无人类玩家校准；必胜机器验证不等于难度验证。

- 旧189题TabPFN开发组隔离CV准确率58.64%，4分召回0；没有达到90%，没有新的独立四分类测试结果。

- 固定验证集扩充到334题：RF57.53%→43.90%，MLP51.10%→45.37%，交叉注意力52.90%→42.45%，TabPFN58.64%→39.22%。新增数据没有提升，原因尚未确证。

- 题族、来源、父题和配对按连通组隔离。重复开发使用的验证集不能视为独立最终测试。

- 未训练或接入用户另一台电脑上的生成器。没有可宣称可靠的生成器奖励模型。

## 目录

- research/data300_v1：334题、证据、争议隔离、拒绝样本、标签和MODEL_RESULTS.md。

- research/scorer_data300_v1、scorer_data300_tabpfn：扩充对照、逐题预测、权重/分组审计、环境和结果。

- research/scorer_v6、scorer_v6_tabpfn：交叉熵四分类与argmax推理、冻结CV、模型。

- research/mechanism_pairs_v1、v2：机制配对及正反例。strict_holdout_v6只有候选池，不是合格四分类验收集。

- 其他research目录保留历史实验、失败结果和生成验证脚本。

建议先人工校准旧3分和新2分尺度、补独立4分题族，再冻结新测试集。不要把同族变体当独立样本。

## 恢复

最终Release final-handoff-20261105 的 project-snapshot.tar.gz 包含仓库工作树、Git忽略的两个TabPFN权重、隔离Python依赖及来源抓取材料。SHA256SUMS校验压缩包；ARCHIVE_MANIFEST.json校验逐文件。无.git、缓存、密钥或共享Conda环境。

解压后得到doudizhu-endgame-generator目录。要保留Git历史，另行clone仓库，再合并解压内容。归档中的runtime包含平台相关文件，新平台建议重装。

原环境Linux/Python3.11，torch2.5.1+cu121、numpy1.26.4、sklearn1.9.0。环境完整清单见ARCHIVE_ENVIRONMENT.json。示例命令：

    python3.11 -m venv .venv

    . .venv/bin/activate

    pip install torch==2.5.1 numpy==1.26.4 scikit-learn==1.9.0 joblib

    pip install --target research/scorer_v2_prior_trial/runtime --upgrade --no-deps tabpfn==2.0.9 scikit-learn==1.6.1 einops==0.8.1 huggingface-hub==0.28.1 requests==2.32.3

    python tools/relocate.py --apply

    python research/data300_v1/tests.py

    python research/scorer_v6/tests.py

    python research/scorer_data300_v1/tests.py

上述安装在其他系统可能需补足平台依赖；恢复验证复用已有基础Python环境，不代表所有平台冷安装均已验证。relocate仅替换历史Python入口的原服务器绝对路径，数据/报告中的历史路径保留。

## 推理与复训

    python research/scorer_v6_tabpfn/predict.py input.json

    python research/scorer_data300_v1/compare.py

    python research/scorer_data300_tabpfn/compare.py

输入为包含hands、turn、last、passes的牌局对象；编码和结构参见数据input字段与牌型模块。输出四类概率及argmax，不作回归取整。训练预算与配置见各目录README和脚本。

## 第三方材料

TabPFN官方来源与许可： https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt 、 https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt 。下载URL与哈希见模型元数据。第三方模型、代码、抓取材料沿用各自许可，不因本库公开改变。

恢复测试见RESTORE_VERIFICATION.md。