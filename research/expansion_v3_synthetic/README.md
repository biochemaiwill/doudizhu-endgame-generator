# 第二轮独立结构候选
固定种子2026110513，三家4—10张、总剩余不超过24、牌点集中5—7种。最多6000次尝试、480秒、100个认证候选。筛选条件不是难度标签。
formal_candidates.jsonl 当前17道人工试标3分题，已逐题阅读完整根动作结果和保留的错误续着。没有标4或5分；不为数量抬分。
annotation-validation.json 是对这17道从序列化牌面重新执行三路根动作核对的通过记录。胜负验证不证明人类难度准确，标签尚无玩家校准。
search-ledger.jsonl.gz 保留每次成功、失败与超时；verified-pool.json 和 run.json 在采样结束后生成。未标注认证候选不自动进入正式数据。
去重覆盖pilot100、highscore_candidates_v1、expansion_v2已有结构化文件及启动时grade5_search_v3已存在候选；后续并行新增仍由总装者交叉去重。
与真实完整发牌历史无关，是合成剩余手牌局面。同一生成器独立采样的deal family不是独立采集来源。
未修改旧数据、训练或划分，无额外依赖与模型下载。运行：python3 research/expansion_v3_synthetic/mine.py（拒绝覆盖旧日志）。
