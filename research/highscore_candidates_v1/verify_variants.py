import sys,json,time,signal,resource,platform
from pathlib import Path
BASE=Path('/home/wutianxiang/doudizhu-endgame-generator')
sys.path.insert(0,str(BASE/'research/pilot100'))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/'research/highscore_candidates_v1'
resource.setrlimit(resource.RLIMIT_CPU,(265,270))
old=json.loads((OUT/'inputs.json').read_text())[0];old['retry']=True
inputs=[old]
data={r['id']:r for r in map(json.loads,(BASE/'research/pilot100/dataset.jsonl').read_text().splitlines())}
for did,player,remove,add in [('DZ094',0,7,None),('DZ094',0,9,None),('DZ094',1,5,4),('DZ095',0,14,None),('DZ095',1,12,11),('DZ098',0,None,6),('DZ098',2,4,6),('DZ099',0,7,9)]:
 d=data[did];hands=[list(h) for h in d['input']['hands']]
 if remove is not None:hands[player].remove(remove)
 if add is not None:hands[player].append(add)
 inputs.append(dict(source_id=f'{did}-variant-{len(inputs)}',hands=[hand_to_text(hand_from_cards(h)) for h in hands],turn=d['input']['turn'],last_text=None,passes=0,source='synthetic_variant_inspired_by_public_puzzle',family=d['provenance'].get('family',did),parent_id=did,source_url=d['provenance'].get('url'),mutation=dict(player=player,remove=remove,add=add)))
(OUT/'variant-inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
blocked={r['verification']['key'] for r in data.values()};out=[];start=time.process_time()
for r in inputs:
 ts=time.process_time()
 try:
  p=restore(r);validate_position(p);r['key']=state_key(p)
  if r['key'] in blocked:r['status']='duplicate'
  else:
   blocked.add(r['key']);signal.setitimer(signal.ITIMER_REAL,45 if r.get('retry') else 22)
   s=Solver(node_limit=3000000 if r.get('retry') else 1200000);vals=s.move_values(p)
   r['moves']=[move_record(m,w) for m,w in vals];r['winning_moves']=[m.text() for m,w in vals if w]
   ref.solve.cache_clear()
   assert ref.move_values(p)==[((kind,cards),w) for kind,cards,w in signature(vals)]
   assert signature(Solver(node_limit=3000000,canonical=False).move_values(p))==signature(vals)
   r['reference_states']=ref.solve.cache_info().currsize
   r['pv']=trace(p,s,maxplies=60)
   r['trap_examples']=[dict(first=m.text(),continuation=trace(p.play(m),s,maxplies=60)) for m,w in vals if not w][:3]
   r['status']='verified' if r['winning_moves'] else 'losing'
 except (TimeBudget,BudgetExceeded) as e:r['status']='unknown_budget';r['error']=type(e).__name__
 except Exception as e:r['status']='error';r['error']=repr(e)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r['cpu_seconds']=time.process_time()-ts;out.append(r)
 (OUT/'variant-verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(r['source_id'],r['status'],r.get('winning_moves'),flush=True)
(OUT/'variant-run.json').write_text(json.dumps(dict(cpu_seconds=time.process_time()-start,python=sys.version,platform=platform.platform(),random_seed=None,statuses=dict(Counter(r['status'] for r in out))),indent=2))
