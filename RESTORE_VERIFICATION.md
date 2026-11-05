# 归档恢复验证（2026-11-05）

归档对应源码提交987277b。Release下载后验证，而非仅验证上传前副本。

- project-snapshot.tar.gz：104992231字节；SHA256 3cd23e00efdede518acbf83a4296209eeba998a5fd48ac3b27e6cfd6dfdffe45。
- 恢复路径 /home/wutianxiang/doudizhu-restore-check/doudizhu-endgame-generator；逐项核对ARCHIVE_MANIFEST.json的1523文件大小和SHA256全部通过。
- 执行 tools/relocate.py --apply 后，research/data300_v1/tests.py 4项、research/scorer_v6/tests.py 3项、research/scorer_data300_v1/tests.py 3项全部通过。
- 执行 research/scorer_v6_tabpfn/predict.py 对归档数据中的一个input推理成功；类别[1,2,3,4]、概率和为1、argmax为2。
- 复用原服务器基础Python3.11/torch环境；隔离TabPFN运行时及权重来自下载归档。没有重新训练全部模型，也未验证Windows环境冷安装。
- 源码、数据、旧实验结果和被忽略的研究产物已归档；共享Conda、代理与其他项目不在删除或归档范围。
- 此恢复记录是快照上传后的补充文档，以Git仓库最新版为准，快照包含的主体研究文件未修改。
