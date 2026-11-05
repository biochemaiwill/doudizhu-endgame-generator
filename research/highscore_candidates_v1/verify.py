import sys,json,time,signal,resource,platform
from pathlib import Path
BASE=Path('/home/wutianxiang/doudizhu-endgame-generator')
sys.path.insert(0,str(BASE/'research/pilot100'))
from build import *
from verify_sources import restore
import reference_ordered as ref
OUT=BASE/'research/highscore_candidates_v1'
resource.setrlimit(resource.RLIMIT_CPU,(290,295))
inputs=[
dict(source_id='33iq-507273',hands=['2QJJT98766543','DAA98875544','433'],turn=0,last_text=None,passes=0,source_url='https://www.33iq.com/topic/574/7/',author='katy Clarkson',source_question='507273',source_note='Public topic search snapshot confirms all hands; previously attempted but not included in pilot100.'),
dict(source_id='33iq-579570',hands=['2K','XAKJTT99884','22T98665443'],turn=2,last_text='5',passes=0,source_url='https://www.33iq.com/topic/2425/',author='smok',source_question='579570',source_note='Public topic lists hands then lower farmer plays 5. One 5 removed from listed XAKJTT998854; interpretation requires image confirmation.')]
(OUT/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
blocked={json.loads(l)['verification']['key'] for l in (BASE/'research/pilot100/dataset.jsonl').read_text().splitlines()}
out=[];start=time.process_time()
for row in inputs:
 r=dict(row);ts=time.process_time()
 try:
  p=restore(r);validate_position(p);r['key']=state_key(p)
  if r['key'] in blocked:r['status']='duplicate'
  else:
   signal.setitimer(signal.ITIMER_REAL,90)
   s=Solver(node_limit=900000);vals=s.move_values(p)
   r['moves']=[move_record(m,w) for m,w in vals];r['winning_moves']=[m.text() for m,w in vals if w]
   ref.solve.cache_clear();rv=ref.move_values(p)
   assert rv==[((kind,cards),w) for kind,cards,w in signature(vals)],'reference mismatch'
   assert signature(Solver(node_limit=900000,canonical=False).move_values(p))==signature(vals),'canonical mismatch'
   r['reference_states']=ref.solve.cache_info().currsize
   r['pv']=trace(p,s);r['trap_examples']=[dict(first=m.text(),continuation=trace(p.play(m),s)) for m,w in vals if not w][:4]
   r['status']='verified' if r['winning_moves'] else 'losing'
 except (TimeBudget,BudgetExceeded) as e:r['status']='unknown_budget';r['error']=type(e).__name__
 except Exception as e:r['status']='error';r['error']=repr(e)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r['cpu_seconds']=time.process_time()-ts;out.append(r)
 (OUT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(r['source_id'],r['status'],r.get('winning_moves'),flush=True)
(OUT/'run.json').write_text(json.dumps(dict(cpu_seconds=time.process_time()-start,python=sys.version,platform=platform.platform(),random_seed=None,mode='deterministic public positions',blocked_keys=len(blocked),command='PYTHONDONTWRITEBYTECODE=1 python3 research/highscore_candidates_v1/verify.py',statuses=dict(Counter(r['status'] for r in out))),indent=2))
