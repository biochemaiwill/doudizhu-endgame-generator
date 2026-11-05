import os,sys,json,gzip,random,signal,time,copy
from pathlib import Path
import numpy as np,torch
from torch import nn
B=Path('/home/wutianxiang/doudizhu-endgame-generator');R=B/'research/scorer_v5'
sys.path.insert(0,str(B/'research/scorer_v4'))
from train import Net,metrics,encode_structured,class_weights
sys.path.insert(0,str(B/'research/pilot100'))
from build import Position,Solver,signature,state_key,hand_from_cards,hand_to_cards,full_deck,TimeBudget,BudgetExceeded
import reference_ordered as ref
torch.set_num_threads(1)
def save(n,x): (R/n).write_text(json.dumps(x,indent=2))
def features(p):
 return encode_structured(dict(hands=[hand_to_cards(h) for h in p.hands],turn=p.turn,last=None,passes=0))
def action(m):
 v=np.zeros(32,np.float32);v[int(m.type)]=1;v[15]=m.main/17;v[16]=m.length/12
 for c in m.cards:v[17+c-3]+=.25
 return v
def run():
 if (R/'results.json').exists():raise SystemExit('frozen')
 rng=random.Random(2026110518);blocked=set()
 for f in [B/'research/scorer_v4/dataset.jsonl']:
  for s in f.read_text().splitlines():blocked.add(json.loads(s)['verification']['key'])
 xs=[];acts=[];ys=[];groups=[];ledger=[];start=time.time();n=0
 while n<1000 and time.time()-start<300:
  deck=full_deck();rng.shuffle(deck);sizes=[rng.randint(3,6) for _ in range(3)];hs=[];k=0
  for size in sizes:hs.append(hand_from_cards(deck[k:k+size]));k+=size
  p=Position(tuple(hs),rng.randrange(3));key=state_key(p)
  if key in blocked:continue
  blocked.add(key)
  try:
   signal.setitimer(signal.ITIMER_REAL,1)
   vals=Solver(node_limit=150000).move_values(p)
   assert [((t,c),w) for t,c,w in signature(vals)]==ref.move_values(p)
   x=features(p)
   for m,w in vals:xs.append(x);acts.append(action(m));ys.append(float(w));groups.append(n)
   ledger.append(dict(key=key,input=x.tolist(),actions=[action(m).tolist() for m,w in vals],labels=[int(w) for m,w in vals]));n+=1
  except (TimeBudget,BudgetExceeded):pass
  finally:signal.setitimer(signal.ITIMER_REAL,0)
 with gzip.open(R/'pretrain.json.gz','wt') as f:json.dump(ledger,f)
 order=list(range(n));rng.shuffle(order);valid=set(order[:n//5]);ai=np.array([i for i,g in enumerate(groups) if g not in valid]);av=np.array([i for i,g in enumerate(groups) if g in valid])
 X=torch.tensor(np.array(xs));A=torch.tensor(np.array(acts));Y=torch.tensor(ys);aux=[]
 rows=[json.loads(s) for s in (B/'research/scorer_v4/dataset.jsonl').read_text().splitlines()];split=json.loads((B/'research/scorer_v4/split.json').read_text());imap={r['id']:i for i,r in enumerate(rows)};ix={k:np.array([imap[i] for i in v]) for k,v in split['ids'].items()}
 dx=torch.tensor(np.stack([encode_structured(r['input']) for r in rows]));dy=np.array([[r['target'][k] for k in ['overall','reasoning_depth','move_concealment']] for r in rows],np.float32);target=torch.tensor((dy[:,:,None]>np.arange(1,4)[None,None]).astype(np.float32))
 predictions={};arts={};val={};logs={}
 for seed in [17,41,73]:
  torch.manual_seed(seed);net=Net('cross');head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1));capt=[]
  def hook(mod,inp):capt.append(inp[0][:,:64])
  handle=net.head.register_forward_pre_hook(hook)
  params=[p for name,p in net.named_parameters() if not name.startswith(('head.','base','steps'))]+list(head.parameters());opt=torch.optim.AdamW(params,lr=.001)
  for ep in range(20):
   net.train();head.train()
   for batch in np.array_split(np.random.default_rng(seed+ep).permutation(ai),max(1,len(ai)//128)):
    capt.clear();net(X[batch]);z=head(torch.cat([capt[-1],A[batch]],1)).squeeze(1);loss=nn.functional.binary_cross_entropy_with_logits(z,Y[batch]);opt.zero_grad();loss.backward();opt.step()
  net.eval();head.eval()
  with torch.no_grad():
   capt.clear();net(X[av]);prob=torch.sigmoid(head(torch.cat([capt[-1],A[av]],1)).squeeze(1));pred=prob>=.5;y=Y[av].bool()
   aux.append(dict(seed=seed,accuracy=float((pred==y).float().mean()),balanced_accuracy=float(((pred[y]==y[y]).float().mean()+(pred[~y]==y[~y]).float().mean())/2)))
  handle.remove();pre={k:v.clone() for k,v in net.state_dict().items() if not k.startswith(('head.','base','steps'))}
  for arm in ['scratch','pretrained']:
   torch.manual_seed(seed);m=Net('cross')
   if arm=='pretrained':m.load_state_dict(pre,strict=False)
   opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);tr,vi=ix['train'],ix['val'];w=torch.tensor(class_weights(dy[tr,0]));best=None;bk=None;be=0
   for ep in range(1,251):
    m.train();opt.zero_grad();l=(nn.functional.binary_cross_entropy_with_logits(m(dx[tr]),target[tr],reduction='none').mean((1,2))*w).sum()/w.sum();l.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval()
    with torch.no_grad():pv=m.scores(dx[vi]).numpy()
    met=metrics(dy[vi],pv);key=(met['accuracy'],met['macro_recall'],-met['mae'])
    if bk is None or key>bk:bk=key;be=ep;best=copy.deepcopy(m.state_dict())
    if ep-be>=35:break
   m.load_state_dict(best);m.eval()
   with torch.no_grad():p=m.scores(dx).numpy()
   predictions.setdefault(arm,[]).append(p);arts.setdefault(arm,[]).append({k:v.tolist() for k,v in best.items()});logs.setdefault(arm,[]).append(dict(seed=seed,best_epoch=be,stop=ep));print(seed,arm,bk,flush=True)
 for arm,p in predictions.items():val[arm]=metrics(dy[ix['val']],np.mean(p,0)[ix['val']])
 selected=max(val,key=lambda k:(val[k]['accuracy'],val[k]['macro_recall'],-val[k]['mae']))
 save('selection.json',dict(selected=selected,validation=val,test_used=False))
 result=dict(pretrain_states=n,pretrain_actions=len(ys),pretrain_validation=aux,validation=val,test={k:metrics(dy[ix['test']],np.mean(v,0)[ix['test']]) for k,v in predictions.items()},history=logs,selected=selected,seconds=time.time()-start,limitations='same previously exposed development split; no independent human calibration')
 save('results.json',result)
 with gzip.open(R/'models.json.gz','wt') as f:json.dump(arts,f)
 save('predictions.json',{arm:np.mean(p,0).tolist() for arm,p in predictions.items()});print(json.dumps(result),flush=True)
if __name__=='__main__':run()
