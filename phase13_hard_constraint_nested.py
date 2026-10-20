#!/usr/bin/env python3
"""CHEM-02 Phase 13: retrospective grouped nested comparison of truly hard-gated van't Hoff vs tuned direct ML.
Original file checksum and metadata provenance checked. Test groups excluded from all fit and selection.
No independent prospective test; one internal validation; not evidence of unqualified physical consistency.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
import argparse,hashlib,json,time,platform
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem, DataStructs, RDLogger,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from lightgbm import LGBMRegressor
import lightgbm
RDLogger.DisableLog('rdApp.*')
ap=argparse.ArgumentParser()
ap.add_argument('--metadata',default='/mnt/data/CHEM02_REAL_RESULTS/harmonized_metadata.csv')
ap.add_argument('--raw',default='/mnt/data/MixtureSolDB.csv')
ap.add_argument('--out',default='/mnt/data/CHEM02_PHASE13')
ap.add_argument('--splits',default='source_disjoint:1001,joint_chemical_OOD:1001')
ap.add_argument('--threads',type=int,default=3)
args=ap.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
assert hashlib.md5(Path(args.raw).read_bytes()).hexdigest()=='597e31685fe36c78d568f3d37b6e1336'
d=pd.read_csv(args.metadata); assert len(d)==172880 and d.RecordKey.is_unique
assert np.all(d.y<=0) and np.all(d.z.between(0,1)) and np.all((d.A+' || '+d.B)==d.pair)
valid=d.source_count.to_numpy()==1; nonpure=(~d.endpoint.to_numpy(bool)); idx=np.flatnonzero(valid)
sol=d.solute.to_numpy(); pair=d.pair.to_numpy(); src=d.source.to_numpy(); system=d.system.to_numpy(); A=d.A.to_numpy(); B=d.B.to_numpy()
T=d.T.to_numpy() if False else d['T'].to_numpy(np.float64)
y=d.y.to_numpy(np.float64); z=d.z.to_numpy(np.float64); u=(1000/T-1000/298.15).astype(np.float32)
# Stable descriptor representations computed only from structures (no labels)
smiles=sorted(set(sol)|set(A)|set(B));fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=128);emb={}
for s in smiles:
 m=Chem.MolFromSmiles(s)
 if m is None: raise ValueError(f'invalid molecular representation: {s}')
 f=np.zeros(128,np.float32);DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(m),f)
 desc=np.asarray([Descriptors.MolWt(m)/500,Descriptors.MolLogP(m)/10,Descriptors.TPSA(m)/200,Descriptors.NumHDonors(m)/10,Descriptors.NumHAcceptors(m)/10],np.float32)
 emb[s]=np.r_[f,desc]
def mt(v): return np.stack([emb[s] for s in v]).astype(np.float32)
S=mt(sol); EA=mt(A); EB=mt(B); mix=z[:,None]*EA+(1-z)[:,None]*EB;g=(z*(1-z)).astype(np.float32)
X0=np.column_stack((S,mix,abs(EA-EB),S[:,:128]*mix[:,:128],z,g)).astype(np.float32)
Xd=np.column_stack((X0,T/300,300/T)).astype(np.float32)
XpA=np.column_stack((S,EA)).astype(np.float32);XpB=np.column_stack((S,EB)).astype(np.float32)
# Group splits coincide with definitions of Phase 12, new seeds.
def split(base,seed,kind):
 rng=np.random.default_rng(seed)
 if kind=='source_disjoint':
  choices=sorted(set(src[base]));held=set(rng.choice(choices,size=max(1,int(.20*len(choices))),replace=False));tm=np.isin(src[base],list(held))
  return base[~tm],base[tm&nonpure[base]]
 if kind=='joint_chemical_OOD':
  sl=sorted(set(sol[base]));pa=sorted(set(pair[base]));ss=set(rng.choice(sl,size=max(1,int(.22*len(sl))),replace=False));pp=set(rng.choice(pa,size=max(1,int(.22*len(pa))),replace=False));sm=np.isin(sol[base],list(ss));pm=np.isin(pair[base],list(pp))
  return base[~(sm|pm)],base[sm&pm&nonpure[base]]
 raise ValueError(kind)
def audit(fit,test,kind):
 assert not set(fit)&set(test)
 if kind=='source_disjoint': assert not set(src[fit])&set(src[test])
 else:
  assert not set(sol[fit])&set(sol[test]); assert not set(pair[fit])&set(pair[test])
def macro(y_true,pr,ix):
 err=np.abs(y_true-pr)
 return float(pd.DataFrame({'s':system[ix],'e':err}).groupby('s',sort=False).e.mean().mean())
def lgb(seed,conf):
 return LGBMRegressor(n_estimators=conf['trees'],num_leaves=conf['leaves'],reg_lambda=conf['l2'],learning_rate=.06,min_child_samples=35,
                     colsample_bytree=.85,subsample=.85,verbosity=-1,random_state=seed,n_jobs=args.threads)
# Preregistered bounded search space; budget/capacity not identical between multihead and direct.
C=[{'trees':250,'leaves':31,'l2':4.},{'trees':600,'leaves':63,'l2':8.}]
# Eligible temperature curves fitted solely on selected training indices (source-specific, >=3 T, span >=8 K).
def fit_curves(train):
 t0=time.monotonic();sub=d.iloc[train][['solute','pair','z_key','source','pure_solvent','endpoint']].copy();sub['rid']=train
 ids=[];a=[];b=[];is_pure=[];pures=[];ss=[]
 for _,grp in sub.groupby(['solute','pair','z_key','source'],sort=False,dropna=False):
  ix=grp.rid.to_numpy(dtype=np.int64)
  if len(ix)<3 or len(np.unique(T[ix]))<3 or T[ix].max()-T[ix].min()<8:continue
  uu=u[ix].astype(float);yy=y[ix]; uu0=uu-uu.mean(); denom=np.dot(uu0,uu0)
  if denom<1e-10:continue
  bb=np.dot(uu0,yy-yy.mean())/denom;aa=yy.mean()-bb*uu.mean()
  if not (np.isfinite(aa) and np.isfinite(bb) and -20<=aa<=1 and -30<=bb<=30):continue
  ids.append(ix[0]);a.append(aa);b.append(bb);is_pure.append(bool(grp.endpoint.iloc[0]));pures.append(grp.pure_solvent.iloc[0]);ss.append(grp.source.iloc[0])
 res=pd.DataFrame({'ix':ids,'a':a,'b':b,'pure':is_pure,'pure_solvent':pures,'source':ss})
 print('CURVES',len(train),len(res),'pure',int(res.pure.sum()),'seconds',round(time.monotonic()-t0,1),flush=True)
 return res

def gate_fit(train,seed,conf):
 cf=fit_curves(train)
 ci=cf.ix.to_numpy(dtype=int);pure=cf.pure.to_numpy(bool)
 pidx=ci[pure]; ic=ci[~pure]
 if len(pidx)<50 or len(ic)<500:raise ValueError('insufficient temporal curves')
 # Pure molecular pair features; source-specific pure anchor coefficients dedup to avoid multipair repetition.
 pA=z[pidx]>.5
 temp=pd.DataFrame({'solute':sol[pidx], 'solvent':cf.loc[pure,'pure_solvent'].to_numpy(), 'source':cf.loc[pure,'source'].to_numpy(),
                    'a':cf.loc[pure,'a'].to_numpy(),'b':cf.loc[pure,'b'].to_numpy(),'ref':np.arange(len(pidx))})
 ag=temp.groupby(['solute','solvent','source'],sort=False).agg(a=('a','median'),b=('b','median'),ref=('ref','first')).reset_index()
 ref=ag.ref.to_numpy(dtype=int);xpure=np.where(pA[:,None],XpA[pidx],XpB[pidx])[ref]
 pa=lgb(seed+21,conf);pb=lgb(seed+22,conf)
 pa.fit(xpure,ag.a.to_numpy());pb.fit(xpure,ag.b.to_numpy())
 gg=g[ic];i_a=cf.loc[~pure,'a'].to_numpy(); i_b=cf.loc[~pure,'b'].to_numpy()
 b0a=z[ic]*pa.predict(XpA[ic])+(1-z[ic])*pa.predict(XpB[ic]);b0b=z[ic]*pb.predict(XpA[ic])+(1-z[ic])*pb.predict(XpB[ic])
 ra_target=(i_a-b0a)/np.maximum(gg,1e-4);rb_target=(i_b-b0b)/np.maximum(gg,1e-4)
 weights=(gg/.25)**2
 ra=lgb(seed+31,conf);rb=lgb(seed+32,conf)
 ra.fit(X0[ic],ra_target,sample_weight=weights);rb.fit(X0[ic],rb_target,sample_weight=weights)
 def pred(ix):
  pred_a=z[ix]*pa.predict(XpA[ix])+(1-z[ix])*pa.predict(XpB[ix])+g[ix]*ra.predict(X0[ix]);
  pred_b=z[ix]*pb.predict(XpA[ix])+(1-z[ix])*pb.predict(XpB[ix])+g[ix]*rb.predict(X0[ix]);
  return np.minimum(pred_a+u[ix]*pred_b,0.)
 return pred,ci,{'n_curves':len(cf),'n_pure_anchors':len(ag),'n_interior_curves':len(ic)}

results=[];inner_results=[];bootstrap=[];audits=[]
for spec in args.splits.split(','):
 kind,seed_s=spec.split(':');seed=int(seed_s)
 start=time.monotonic();outer_fit,outer_test=split(idx,seed,kind); inner_fit,inner_val=split(outer_fit,seed+8713,kind)
 audit(outer_fit,outer_test,kind);audit(inner_fit,inner_val,kind)
 assert len(inner_val)>500 and len(outer_test)>500
 aud={'protocol':kind,'seed':seed,'outer_fit':len(outer_fit),'outer_test':len(outer_test),'inner_fit':len(inner_fit),'inner_val':len(inner_val),
      'n_test_systems':len(set(system[outer_test])), 'overlap_sources':len(set(src[outer_fit])&set(src[outer_test])),
      'overlap_solutes':len(set(sol[outer_fit])&set(sol[outer_test])), 'overlap_pairs':len(set(pair[outer_fit])&set(pair[outer_test]))}
 audits.append(aud);pd.DataFrame(audits).to_csv(O/'split_audit.csv',index=False)
 print('SPLIT',aud,flush=True)
 best={}
 # Tuning model by inner system-macro MAE (never outer test values)
 for method in ['direct_all','direct_curve_matched','gated_vanthoff']:
  best_score=float('inf');best_conf=None
  for conf in C:
   if method=='gated_vanthoff':
    fn,curveix,ct=gate_fit(inner_fit,seed+777,conf);p=fn(inner_val);ntrain=len(curveix)
   else:
    if method=='direct_curve_matched':
     cx=fit_curves(inner_fit).ix.to_numpy(dtype=int); # same curves used for gated coefficients
     mask=np.isin(inner_fit,d.iloc[cx][['solute','pair','z_key','source']].drop_duplicates().merge(d.iloc[inner_fit][['solute','pair','z_key','source']],how='right',indicator=True)['source'].dropna().to_numpy()) if False else None
     # supervised observations restricted to qualifying curves, identified via group keys.
     keys=d.iloc[cx][['solute','pair','z_key','source']].drop_duplicates();frame=d.iloc[inner_fit][['solute','pair','z_key','source']].copy();frame['_rid']=inner_fit
     cur=frame.merge(keys,on=['solute','pair','z_key','source'],how='inner',sort=False);fit_ix=cur['_rid'].to_numpy(dtype=int)
    else: fit_ix=inner_fit
    m=lgb(seed+777,conf);m.fit(Xd[fit_ix],y[fit_ix]);p=np.minimum(m.predict(Xd[inner_val]),0.);ntrain=len(fit_ix)
   score=macro(y[inner_val],p,inner_val)
   inner_results.append({**aud,'method':method,**conf,'inner_macro_MAE':score,'n_supervised_rows_or_curves':ntrain})
   pd.DataFrame(inner_results).to_csv(O/'inner_search.csv',index=False)
   print('INNER',kind,seed,method,conf,'macro',round(score,5),flush=True)
   if score<best_score:best_score=score;best_conf=conf.copy()
  best[method]=(best_conf,best_score)
 # Fit on full outer training, evaluate on outer test one time
 preds={}
 for method,(conf,score) in best.items():
  if method=='gated_vanthoff':
   fn,cx,ct=gate_fit(outer_fit,seed+79,conf);p=fn(outer_test);train_n=len(cx)
  else:
   if method=='direct_curve_matched':
    cx=fit_curves(outer_fit).ix.to_numpy(dtype=int);keys=d.iloc[cx][['solute','pair','z_key','source']].drop_duplicates()
    frame=d.iloc[outer_fit][['solute','pair','z_key','source']].copy();frame['_rid']=outer_fit
    train_ix=frame.merge(keys,on=['solute','pair','z_key','source'],how='inner',sort=False)['_rid'].to_numpy(dtype=int)
   else: train_ix=outer_fit
   m=lgb(seed+79,conf);m.fit(Xd[train_ix],y[train_ix]);p=np.minimum(m.predict(Xd[outer_test]),0.);train_n=len(train_ix)
  preds[method]=p
  met={'protocol':kind,'seed':seed,'method':method,**conf,'inner_macro_MAE':score,'test_system_macro_MAE':macro(y[outer_test],p,outer_test),
       'test_micro_MAE':float(np.mean(abs(y[outer_test]-p))), 'test_RMSE':float(np.sqrt(np.mean((y[outer_test]-p)**2))),
       'n_train_supervision':train_n,'test_n':len(outer_test),'n_test_systems':len(set(system[outer_test]))}
  results.append(met);pd.DataFrame(results).to_csv(O/'outer_metrics.csv',index=False)
  print('OUTER',met,flush=True)
 # paired differences macro error, test group bootstrapping
 er=pd.DataFrame({'system':system[outer_test], **{m:abs(y[outer_test]-p) for m,p in preds.items()}}).groupby('system',sort=False).mean()
 rng=np.random.default_rng(seed+90482)
 for ref in ('direct_all','direct_curve_matched'):
  delta=(er['gated_vanthoff']-er[ref]).to_numpy();boot=np.asarray([delta[rng.integers(0,len(delta),size=len(delta))].mean() for i in range(1000)])
  bootstrap.append({'protocol':kind,'seed':seed,'comparison':'gated_vanthoff_minus_'+ref,'delta_system_macro':float(delta.mean()),
                   '95CI_lower':float(np.quantile(boot,.025)),'95CI_upper':float(np.quantile(boot,.975)),'n_systems':len(delta)})
 pd.DataFrame(bootstrap).to_csv(O/'paired_cluster_bootstrap.csv',index=False)
 # Store local rowwise predictions (not redistributing dataset) for post-hoc audits
 pd.DataFrame({'RecordKey':d.iloc[outer_test].RecordKey.to_numpy(),'system':system[outer_test],'true_y':y[outer_test],**preds}).to_csv(O/f'LOCAL_predictions_{kind}_{seed}.csv.gz',index=False,compression='gzip')
 print('FINISHED SPLIT',kind,seed,round(time.monotonic()-start,2),'seconds',flush=True)
with open(O/'provenance.json','w') as f:
 json.dump({'dataset_md5':hashlib.md5(Path(args.raw).read_bytes()).hexdigest(), 'metadata_sha256':hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'software':{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'lightgbm':lightgbm.__version__},
 'method_warning':'Retrospective dataset; model gated coefficient fits trained only on temperature-series curves. Gated residual training uses in-sample fitted pure heads (not cross-fitting); outer test data always excluded. Same matched training data comparator also evaluated. Selection uses inner holdout; only one seed per protocol.'},f,indent=2)
print('DONE',flush=True)
