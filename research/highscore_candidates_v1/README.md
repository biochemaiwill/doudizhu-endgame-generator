# Next-version high-score candidates, 2026-11-05

Isolated research only. Frozen pilot100 and current training/splits are unchanged.

## Result
Two new non-isomorphic 4/4/4 candidates, both explicitly synthetic variants inspired by existing public puzzles:
- DZ094-variant-3: partner's 5 replaced by 4; active farmer must lead 888+6.
- DZ098-variant-7: active farmer's 4 replaced by 6; must lead bare TTT.
Keep each in its original parent's family. These are not independent new public puzzles and must never be split across parent/variant families.

candidates.jsonl contains all 11 verified annotations (2 score-4, 2 score-3, 1 score-2 variants; 6 score-1 public game levels).
highscore.jsonl contains only the 2 score-4 candidates.
No score 5. Labels are unblinded single-assistant estimates, not human calibration.

## Public-source gaps
33IQ 507273 was retried at 3,000,000 nodes / 45 s: optimized root found winning 2, but independent certification exceeded time; retain unknown, no label.
579570: one conditional interpretation verified, but source hand timing remains ambiguous; retain quarantine, no label.
579588 and 579914 have incomplete hands in public text; not guessed.
Cyracle public JS contains 100 levels despite page copy saying 30. Six late levels were independently verified; all legal leads win, all grade 1. No hard public-original puzzle certified.

## Evidence
verification.json: initial public attempts, including quarantine.
variant-verification.json: retry and 8 variants, retaining losing and unknown attempts.
cyracle-verification.json: 6 public-original levels.
All certified rows agree for every legal root action between optimized solver, canonical-off solver, and independent ordered reference enumeration. Existing vendor rules remain the scope; no full-game reachability or cross-rules proof.
Canonical keys checked against all 100 frozen positions and within each batch; variants differ structurally, not only by order-preserving rank relabeling.
PV is an illustrative line; losing side may choose arbitrary loss. Do not interpret length as human reasoning depth.
Source URLs, parent family, exact rank mutations and moves are retained in JSON. Raw website HTML saved for reproducibility. No paid model/API, GPU, training, commit or push.

## Budget / reproduction
Initial CPU 9.03 s; variants/retry 49.92 s; public-game verification 0.04 s. All solvers single process, no parallel CPU work. At most 16 unique positions considered (2 initial + 8 variants + 6 public game), plus one retry.
Run from repository root with PYTHONDONTWRITEBYTECODE=1:
python3 research/highscore_candidates_v1/verify.py
python3 research/highscore_candidates_v1/verify_variants.py
python3 research/highscore_candidates_v1/verify_cyracle.py
The retained current artifacts are the delivered evidence; reruns replace their own isolated output files.

## 中文交付说明
新增两道4分候选，均为已有公开4分题启发的变体，必须与父题保持同一数据家族。没有5分。没有回填本轮scorer_v1的train/val/test。
仅提交结构化候选、来源及验证；完整网页HTML/JS和未抽取使用的整套题库不提交，服务器保留原始快照，哈希见manifest.json。verify_cyracle.py使用已保存的6题输入，不依赖完整网页脚本。
