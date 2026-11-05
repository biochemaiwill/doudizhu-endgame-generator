"""Relocate historical Python entrypoints after restoring."""
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument("--apply",action="store_true");a=p.parse_args()
root=Path(__file__).resolve().parents[1]
old="/home/wutianxiang/doudizhu-endgame-generator"
for f in (root/"research").rglob("*.py"):
 if "runtime" in f.parts or "__pycache__" in f.parts: continue
 s=f.read_text()
 if old in s:
  print(f.relative_to(root))
  if a.apply: f.write_text(s.replace(old,str(root)))
