from pathlib import Path
import sys,json,hashlib
B=Path("/home/wutianxiang/doudizhu-endgame-generator");O=B/"research/grade4_expansion_v5"
sys.path.insert(0,str(B/"research/pilot100"))
from build import *
from verify_sources import restore
import reference_ordered as ref
old=list(map(json.loads,(B/"research/scorer_v4/dataset.jsonl").read_text().splitlines()));blocked={x["verification"]["key"] for x in old}
labs={}
for key in ["EX208-s1-4to3","EX208-s1-5to6"]:labs[key]=(4,4,4,"唯一先9，不能先兑现77或KK；后续须按农民A、大王是否接手安排拆77和保留KK，低单改动未移除相互依赖的控制选择。")
labs["EX220-s0-8to6"]=(3,4,3,"先9；地主过时队友只能出J或Q，过或2都输。后续仍有配合，但地主低单降低后清理更直接，保守降为3分。")
for key in ["EX220-s1-4to3","EX220-s2-4to3"]:labs[key]=(4,4,4,"唯一先9；地主过时队友须出J或Q，不能过或先2。之后还要拆JJ并在合适时机让队友2接管，需按地主Q、大王等应对切换。")
for key in ["DZ098-s2-3to6","DZ098-s2-4to7"]:labs[key]=(4,4,4,"只有裸101010获胜，三带一、三带二均输；地主过时队友必须JJJ接手而不能过，地主出222时则须等待后续配合，关键在带牌与牌权转交。")
rows=[];evidence=[]
for r in json.loads((O/"candidate_verification.json").read_text()):
 if r["origin_id"] not in labs:continue
 p=restore(r);validate_position(p);key=state_key(p);assert key not in blocked;blocked.add(key)
 v=Solver(node_limit=3000000).move_values(p)
 assert ref.move_values(p)==[((t,c),w) for t,c,w in signature(v)]
 assert signature(v)==signature(Solver(node_limit=3000000,canonical=False).move_values(p))
 a,d,c,why=labs[r["origin_id"]];ident=f"G4V5-{len(rows)+1:02d}"
 inp=dict(hands=[hand_to_cards(h) for h in p.hands],turn=p.turn,last=None,passes=0,rank_encoding="3..14=3..A;15=2;16=small_joker;17=big_joker",roles=["landlord","farmer_down","farmer_up"])
 rows.append(dict(id=ident,input=inp,target=dict(overall=a,reasoning_depth=d,move_concealment=c),reason=why,annotation=dict(type="assistant_provisional_estimate",rubric="v0.1",human_calibrated=False,blind=False),provenance=dict(source="synthetic_variant",origin_id=r["origin_id"],parent_id=r["parent_id"],family=r["family"],mutation=r["mutation"]),verification=dict(status="verified",key=key,winning_moves=[m.text() for m,w in v if w],legal_move_count=len(v),evidence="candidate_verification.json; key_branches.json")))
 evidence.append(dict(id=ident,three_way_root_outcomes="pass"))
(O/"dataset.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
(O/"annotations.json").write_text(json.dumps(labs,ensure_ascii=False,indent=2))
(O/"validation.json").write_text(json.dumps(dict(rows=len(rows),grades=dict(Counter(x["target"]["overall"] for x in rows)),checks=evidence,sha256=hashlib.sha256((O/"dataset.jsonl").read_bytes()).hexdigest()),indent=2))
(O/"README.md").write_text("# 高分变体试标\n\n正式7道：6道4分、1道3分。全部与现有182题不重复，并从序列化输入重新三路逐根验证。保留原v0.1口径；分数未经真人校准。\n\n6道4分只涉及3个既有父题家族（EX208、EX220、DZ098），不是6个独立新题族。全部必须与父题同分组，不允许跨train/val/test；本批未加入正在运行的scorer_v5。EX220地主8改6的版本保守降为3分。\n\n正式数据dataset.jsonl，标签annotations.json，全部尝试和反例candidate_verification.json，关键回应key_branches.json，重新认证validation.json。verify_variants.py可重现候选验证，但会覆盖自己的候选文件。finalize.py重建正式数据。\n\n本轮公开比赛来源检索未取得可核对的完整三家牌面，不声称提供职业赛事残局。\n")
print(len(rows),"pass")
