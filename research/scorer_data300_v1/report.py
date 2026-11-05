"""Summarize frozen comparison; no model selection or training here."""
import json
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent;B=R.parent
def main():
 result=json.loads((R/"results.json").read_text());sp=json.loads((R/"split_audit.json").read_text())
 rows=[json.loads(s) for s in (B/"data300_v1/dataset_merged.jsonl").read_text().splitlines()];byid={r["id"]:r for r in rows}
 old=json.loads((B/"scorer_v6/dataset.json").read_text())
 diag={};weight_audit=[]
 for f in sp["folds"]:
  entry=dict(seed=f["seed"],fold=f["fold"],val_n=len(f["val_ids"]),excluded_new=f["excluded_new_related_ids"])
  for arm in ["old","expanded"]:
   ids=f[arm+"_train_ids"];heads={}
   for key in ["overall","reasoning_depth","move_concealment"]:
    c=Counter(byid[i]["target"][key] for i in ids);w={str(g):len(ids)/(4*c[g]) for g in range(1,5)}
    assert all(abs(c[g]*w[str(g)]-len(ids)/4)<1e-8 for g in range(1,5))
    heads[key]={"counts":dict(c),"balanced_weight":w,"total_weight_each_grade":len(ids)/4}
   entry[arm]={"n":len(ids),"heads":heads}
  weight_audit.append(entry)
 lines=["# 扩充至334题后的受控训练对照","",
 "固定旧scorer_v6的六fold验证ID、模型配置和训练预算，只给训练侧加入新题。全部题族/来源/parent/配对连通分组；同验证题族的新题排除。不是独立最终测试，不作90%验收。","",
 "旧train每fold142/143，扩充train286–288；val46/47不变。所有模型保持原配置，MLP/cross实际为191维relational输入。CE三头、argmax；逆频率权重每fold按当前训练标签重新计算，未复用旧权重。","",
 "## 数据分布","", "|数据|1分|2分|3分|4分|总数|","|---|---:|---:|---:|---:|---:|"]
 for label,rr in [("旧",old),("扩充",rows)]:
  c=Counter(r["target"]["overall"] for r in rr);lines.append("|"+label+"|"+"|".join(str(c[g]) for g in range(1,5))+f"|{len(rr)}|")
 lines+=["","新增145条以2分为主，4分题数量未增加。","",
 "## 总体结果","", "|模型|旧准确率|新准确率|变化百分点|旧宏召回|新宏召回|","|---|---:|---:|---:|---:|---:|"]
 for name,m in result.items():
  a,b=m["old"],m["expanded"]
  lines.append(f'|{name}|{100*a["mean_accuracy"]:.2f}%|{100*b["mean_accuracy"]:.2f}%|{100*m["accuracy_delta"]:+.2f}|{100*a["mean_macro_recall"]:.2f}%|{100*b["mean_macro_recall"]:.2f}%|')
 lines+=["","## 分档结果",""]
 for name,m in result.items():
  lines += [f"### {name}","", "|数据|1分召回|2分召回|3分召回|4分召回|","|---|---:|---:|---:|---:|"]
  diag[name]={}
  for arm in ["old","expanded"]:
   v=m[arm];cm=[[sum(f["confusion"][a][b] for f in v["folds"]) for b in range(4)] for a in range(4)]
   diag[name][arm]={"aggregate_confusion":cm,"grade3_predicted2":cm[2][1],"grade3_total":sum(cm[2])}
   lines.append("|"+arm+"|"+"|".join(f"{100*x:.2f}%" for x in v["mean_recall"])+"|")
   lines += ["",arm+" 汇总混淆矩阵（行真值，列预测；两个seed中每题各出现一次，非双倍独立样本）：",json.dumps(cm),""]
 lines += ["## 解读","",
 "这三个固定模型没有得到总体准确率提升。RF的3分召回77.54%降至16.49%，汇总3→2错误25/116增至94/116。",
 "新增样本以2分为主、批次评分口径可能漂移、表示不足都可能有关，本实验不能确认唯一根因。等类权重仅消除总权重差异，不消除类别内部样本分布变化。",
 "结论仅适用于本数据、模型和预算；不能泛化为增加高质量数据无用。新模型仅为实验产物，不替换旧部署。","",
 "## 文件与验证","",
 "- split_audit.json：各fold train/val、新增同族排除、所有组。",
 "- weight_audit.json：旧/新每fold每head标签分布、权重和等类总权重。",
 "- results.json/oof_predictions.json/diagnostics.json：全部指标和可复算预测。",
 "- tests.py：旧验证固定、组隔离、概率和指标复算、旧基线复现。",
 "- compare.py：固定训练实验；report.py：从产物生成本报告；predict.py：仅实验模型推理。",
 "训练命令：/home/wutianxiang/miniconda3/envs/research/bin/python research/scorer_data300_v1/compare.py",
 "测试命令：同Python research/scorer_data300_v1/tests.py",
 "未评估任何新的严格测试集。"]
 (R/"diagnostics.json").write_text(json.dumps(diag,ensure_ascii=False,indent=2)+"\n")
 (R/"weight_audit.json").write_text(json.dumps(weight_audit,ensure_ascii=False,indent=2)+"\n")
 (R/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":main()
