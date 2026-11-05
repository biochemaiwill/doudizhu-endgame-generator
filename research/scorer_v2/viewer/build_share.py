"""Build a self-contained share ZIP without local staging files."""
from pathlib import Path
import io,csv,json,zipfile,hashlib
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def main():
 rows=[json.loads(x) for x in (BASE/"dataset.jsonl").read_text().splitlines()]
 split=json.loads((BASE/"split.json").read_text())
 pairs=json.loads((BASE.parent/"expansion_v2/pairs.json").read_text())
 owners={i:p for p,ids in split["ids"].items() for i in ids}
 symbols={11:"J",12:"Q",13:"K",14:"A",15:"2",16:"小王",17:"大王"}
 def cards(h):return " ".join(str(symbols.get(i,i)) for i in h)
 stream=io.StringIO(newline="")
 writer=csv.writer(stream)
 writer.writerow(["题号","总分","推理深度","出法隐蔽性","地主手牌","地主下家手牌","地主上家手牌","当前出牌者","上一手","连续过牌数","必胜首着","简短依据","数据划分","来源类型","来源编号","来源链接","题族"])
 for r in rows:
  i=r["input"];t=r["target"];p=r["provenance"]
  writer.writerow([r["id"],t["overall"],t["reasoning_depth"],t["move_concealment"],*[cards(h) for h in i["hands"]],["地主","地主下家","地主上家"][i["turn"]],cards(i["last"]["cards"]) if i["last"] else "自由首出",i["passes"],"；".join(r["verification"]["winning_moves"]),r["reason"],owners[r["id"]],p["source"],p["origin_id"],p.get("url") or "",p["family"]])
 readme="""斗地主残局数据集 · 153题分享包
版本：2026-11-05

使用方法
1. 先把本ZIP完整解压到任意文件夹。
2. 双击“斗地主残局数据集_153题.html”，使用浏览器离线打开。
3. 网页支持按分数、来源、数据划分筛选、搜索、展开答案及配对跳转。
不需要安装Python、模型或依赖，不需要服务器连接。打开“原来源”外链时才需要联网。

文件说明
- 斗地主残局数据集_153题.html：人类可读离线数据册，包含全部牌面及答案。
- 斗地主残局数据集_153题.csv：UTF-8 BOM编码表格，可用Excel打开。
- dataset.jsonl：逐行JSON原始标注及来源、验证元信息。
- pairs.json：15组严格等三家牌数对照关系。
- split.json：当前训练/验证/测试的题号及分组记录。
- 文件校验.json：各文件SHA256，用于检查传输完整性。

数据说明
共153题：1分57题、2分46题、3分42题、4分8题，没有5分题。
标签是助手人工试标的弱标签，尚未经过普通玩家解题实验校准。
评分关注人类推理深度与关键出法隐蔽性，不等同于牌数、胜率或搜索节点数。
15组配对的简单一侧是人工构造的一手出完题，不代表自然对局分布。
三人明牌，角色/行动人/上一手及过牌数均需一并使用。
JSON牌点：3..14表示3..A；15表示2；16小王；17大王。
出牌顺序：地主→地主下家→地主上家→地主。
同题族和配对题不得跨train/val/test；split.json保留当前研究划分。
train/val/test为开发研究用途，不是经真人校准的独立最终评测。

代码与更新
https://github.com/biochemaiwill/doudizhu-endgame-generator
来源链接及题号保留于数据中。该分享包不包含训练模型、大型依赖或凭证。
"""
 files={
 "斗地主残局数据集_153题.html":(HERE/"斗地主残局数据集_153题.html").read_bytes(),
 "斗地主残局数据集_153题.csv":stream.getvalue().encode("utf-8-sig"),
 "dataset.jsonl":(BASE/"dataset.jsonl").read_bytes(),
 "pairs.json":(BASE.parent/"expansion_v2/pairs.json").read_bytes(),
 "split.json":(BASE/"split.json").read_bytes(),
 "README_先读我.txt":readme.encode("utf-8")
 }
 hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
 files["文件校验.json"]=json.dumps(hashes,ensure_ascii=False,indent=2).encode("utf-8")
 output=HERE/"斗地主残局数据集_153题_分享包.zip"
 with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,data in files.items():z.writestr(name,data)
 with zipfile.ZipFile(output) as z:
  assert z.testzip() is None
  assert len(z.read("dataset.jsonl").decode().splitlines())==153
  assert len(json.loads(z.read("pairs.json")))==15
  assert len(list(csv.reader(io.StringIO(z.read("斗地主残局数据集_153题.csv").decode("utf-8-sig")))))==154
  for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
 print(json.dumps({"path":str(output),"size":output.stat().st_size,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"html_sha256":hashes["斗地主残局数据集_153题.html"],"entries":list(files)},ensure_ascii=False))
if __name__=="__main__":main()
