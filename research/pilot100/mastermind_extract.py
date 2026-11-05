"""Read small public API pages, retain deduplicated short fully specified states."""
from build import *
import urllib.request,urllib.parse,re,concurrent.futures
from endgame.fast import gen_all
MAP={**{i:i for i in range(3,15)},17:15,20:16,30:17}
def convert(a):return [MAP[x] for x in a]
def parse_row(record):
 text=record["row"]["sentence"]
 if "Assume you are the landlord." not in text:raise ValueError("unsupported acting role")
 ll=json.loads(re.search(r"current cards for the landlord are: (\[[^\]]*\])",text).group(1))
 up=json.loads(re.search(r"current cards for the peasant that plays before landlord are: (\[[^\]]*\])",text).group(1))
 down=json.loads(re.search(r"current cards for the peasant that plays after landlord are: (\[[^\]]*\])",text).group(1))
 lasttwo=json.loads(re.search(r"last two moves by other players are (\[\[.*?\]\])\.",text).group(1))
 # DouZero get_last_two_moves is newest-first; keep this assumption explicit.
 if lasttwo[0]:cards=convert(lasttwo[0]);passes=0
 elif lasttwo[1]:cards=convert(lasttwo[1]);passes=1
 else:cards=[];passes=0
 last=None
 if cards:
  ms=[m for m in gen_all(hand_from_cards(cards)) if tuple(sorted(cards))==m.cards]
  if len(ms)!=1:raise ValueError("ambiguous previous card type")
  last=ms[0]
 pos=Position(tuple(hand_from_cards(convert(h)) for h in (ll,down,up)),0,last,passes)
 validate_position(pos)
 expected=json.loads(re.search(r"legal actions for the landlord are: (\[\[.*?\]\])\.",text).group(1))
 expected={tuple(sorted(convert(x))) for x in expected}
 actual={tuple(m.cards) for m in pos.legal()}
 if expected!=actual:raise ValueError("legal action contract mismatch")
 return pos
def fetch(offset):
 url="https://datasets-server.huggingface.co/rows?"+urllib.parse.urlencode({"dataset":"OpenDILabCommunity/MasterMind","config":"doudizhu","split":"train","offset":offset,"length":100})
 try:return offset,json.load(urllib.request.urlopen(url,timeout=25)),None
 except Exception as e:return offset,None,repr(e)
def main():
 rng=random.Random(20261107)
 offsets=sorted(rng.sample(range(0,570800,100),24))
 rows=[];seen=set();logs=[];counts=Counter()
 for offset,data,error in concurrent.futures.ThreadPoolExecutor(max_workers=3).map(fetch,offsets):
  logs.append({"offset":offset,"error":error,"returned":len(data.get("rows",[])) if data else 0})
  if not data:continue
  for raw in data["rows"]:
   try:
    pos=parse_row(raw);key=state_key(pos)
    if key in seen:counts["duplicate"]+=1;continue
    seen.add(key)
    n=sum(map(hand_size,pos.hands))
    if n>20:counts["too_many_cards"]+=1;continue
    counts["short_candidate"]+=1
    rows.append({"source":"MasterMind train API","source_row":raw["row_idx"],"source_url":"https://huggingface.co/datasets/OpenDILabCommunity/MasterMind","raw_sentence":raw["row"]["sentence"],"key":key,"hands":[hand_to_text(h) for h in pos.hands],"turn":0,"last":move_record(pos.last) if pos.last else None,"passes":pos.passes,"n_cards":n})
   except Exception as e:counts["parse_or_contract_reject"]+=1
  print("PAGE",offset,"candidates",len(rows),"error",error,flush=True)
 (ROOT/"mastermind-candidates.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
 (ROOT/"mastermind-fetch.json").write_text(json.dumps({"offset_seed":20261107,"pages":logs,"counts":dict(counts),"last_two_order":"newest first; based on kwai/DouZero get_last_two_moves, legal moves crosschecked","mapping":{"17":"2","20":"small joker","30":"big joker"}},ensure_ascii=False,indent=2),encoding="utf-8")
 print("FINISH",dict(counts),len(rows),flush=True)
if __name__=="__main__":main()
