#!/usr/bin/env python3
"""Exploratory, post hoc compute-budget-limited triple-disjoint endpoint model.
Three-fold OOF pure-head prediction grouped by solute, inside outer training only.
Neither an external test nor an equal-capacity comparison.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='3'
import time,json,hashlib, warnings, platform, argparse
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
import lightgbm
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
ap=argparse.ArgumentParser()
ap.add_argument('--raw',required=True,type=Path)
ap.add_argument('--metadata',required=True,type=Path)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args()
O=a.out;O.mkdir(exist_ok=True,parents=True)
raw=a.raw; assert hashlib.md5(raw.read_bytes()).hexdigest()=='597e31685fe36c78d568f3d37b6e1336'
d=pd.read_csv(a.metadata)
assert len(d)==172880 and d.RecordKey.is_unique
sol=d.solute.to_numpy();a=d.A.to_numpy();b=d.B.to_numpy();pair=d.pair.to_numpy();src=d.source.to_numpy();system=d.system.to_numpy()
T=d['T'].to_numpy(dtype=np.float32);z=d.z.to_numpy(dtype=np.float32);y=d.y.to_numpy(dtype=np.float32)
end=d.endpoint.to_numpy(dtype=bool);eligible=d.source_count.to_numpy()==1
ids=np.flatnonzero(eligible); g=z*(1-z)
# Smaller but invariant features shared by direct & gated models; no label-dependent transformation.
mo=sorted(set(sol)|set(a)|set(b)); morgan=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=64);emb={}
for s in mo:
 m=Chem.MolFromSmiles(s)
 if m is None:raise ValueError(f'invalid SMILES {s}')
 f=np.zeros(64,np.float32);DataStructs.ConvertToNumpyArray(morgan.GetFingerprint(m),f)
 v=np.array([Descriptors.MolWt(m)/500,Descriptors.MolLogP(m)/10,Descriptors.TPSA(m)/200,Descriptors.NumHDonors(m)/10,Descriptors.NumHAcceptors(m)/10],dtype=np.float32)
 emb[s]=np.r_[f,v]
stack=lambda values:np.stack([emb[s] for s in values])
s=stack(sol);aa=stack(a);bb=stack(b)
mix=z[:,None]*aa+(1-z)[:,None]*bb
Xd=np.column_stack((s,mix,abs(aa-bb),s[:,:64]*mix[:,:64],z,g,T/300,300/T)).astype(np.float32)
# Pure representation uses experimental pure solvent identity not pair-level identity.
pures=d.pure_solvent.to_numpy();xp=np.column_stack((s,stack(np.where(end,pures,a)),T/300,300/T)).astype(np.float32)
# More convenient endpoint features at arbitrary solute/solvent A or B.
xa=np.column_stack((s,aa,T/300,300/T)).astype(np.float32)
xb=np.column_stack((s,bb,T/300,300/T)).astype(np.float32)
model=lambda seed,n: LGBMRegressor(n_estimators=n,num_leaves=31,learning_rate=0.06,min_child_samples=40,reg_lambda=4.0,colsample_bytree=.85,verbosity=-1,n_jobs=3,random_state=seed,max_bin=127)
errors=[];scores=[];strata=[];splits=[];endpoint_scores=[]
def triple(base,seed):
 rng=np.random.default_rng(seed);masks=[]
 for values in [src,sol,pair]:
  uq=sorted(set(values[base]));held=set(rng.choice(uq,size=max(1,int(.30*len(uq))),replace=False));masks.append(np.isin(values[base],list(held)))
 test=masks[0]&masks[1]&masks[2];train=~(masks[0]|masks[1]|masks[2]);return base[train],base[test&~end[base]],base[test&end[base]]
def metrics(ix,p):
 e=abs(y[ix]-p)
 return dict(system_macro_MAE=pd.DataFrame({'s':system[ix],'e':e}).groupby('s').e.mean().mean(),micro_MAE=e.mean(),RMSE=np.sqrt(np.mean((y[ix]-p)**2)))
for seed in (1401,1402,1403):
 st=time.monotonic();train,test,pure_test=triple(ids,seed)
 assert not set(src[train])&set(src[test]); assert not set(sol[train])&set(sol[test]); assert not set(pair[train])&set(pair[test])
 ptr=train[end[train]];mi=train[~end[train]]
 audit={'seed':seed,'train':len(train),'train_pure':len(ptr),'train_interior':len(mi),'test':len(test),'test_pure':len(pure_test),'n_systems':len(set(system[test])),'n_test_sources':len(set(src[test])),'n_test_solutes':len(set(sol[test])),'n_test_pairs':len(set(pair[test])),'shared_sources':0,'shared_solutes':0,'shared_pairs':0}
 splits.append(audit);pd.DataFrame(splits).to_csv(O/'fast_split_audit.csv',index=False)
 print('START',audit,flush=True)
 direct=model(seed,250);direct.fit(Xd[train],y[train]);pdirect=np.minimum(direct.predict(Xd[test]),0.)
 print('DIRECT',seed,metrics(test,pdirect),'elapsed',round(time.monotonic()-st,1),flush=True)
 # Threefold OOF predictions by solute of *training data*. Source/solute/pair test rows not touched.
 usol=sorted(set(sol[train]));rng=np.random.default_rng(seed+455);fold_map={val:i%3 for i,val in enumerate(rng.permutation(usol))}
 fold=np.array([fold_map[v] for v in sol[train]],dtype=np.int8)
 fold_index={rid:f for rid,f in zip(train,fold)}
 ooA=np.full(len(mi),np.nan,dtype=np.float32);ooB=ooA.copy()
 for k in range(3):
  pfit=np.array([rid for rid in ptr if fold_index[rid]!=k],dtype=int)
  vmi=np.flatnonzero(np.array([fold_index[rid]==k for rid in mi]))
  h=model(seed+101+k,150);h.fit(xp[pfit],y[pfit]);ooA[vmi]=h.predict(xa[mi[vmi]]);ooB[vmi]=h.predict(xb[mi[vmi]])
 assert np.isfinite(ooA).all() and np.isfinite(ooB).all()
 head=model(seed+160,150);head.fit(xp[ptr],y[ptr]);
 gate=np.maximum(g[mi],1e-4);interp=z[mi]*ooA+(1-z[mi])*ooB
 res=(y[mi]-interp)/gate;w=(g[mi]/.25)**2
 residual=model(seed+170,150);residual.fit(Xd[mi],res,sample_weight=w)
 pA=head.predict(xa[test]);pB=head.predict(xb[test]);pgated=np.minimum(z[test]*pA+(1-z[test])*pB+g[test]*residual.predict(Xd[test]),0.)
 # Additional diagnostic: do predicted pure endpoints actually match experimental pure anchors?
 if len(pure_test):
  pp_direct=np.minimum(direct.predict(Xd[pure_test]),0.)
  pp_head=np.minimum(head.predict(xp[pure_test]),0.)
  # Group by solute and pure solvent, not the arbitrary pair in which the same endpoint was reported.
  keys=np.char.add(np.char.add(sol[pure_test].astype(str),' || '),pures[pure_test].astype(str))
  p_table=pd.DataFrame({'key':keys,'direct':np.abs(y[pure_test]-pp_direct),'head':np.abs(y[pure_test]-pp_head)}).groupby('key').mean()
  endpoint_scores.append({'seed':seed,'n_pure_points':len(pure_test),'n_pure_solute_solvent':len(p_table),
      'direct_pure_macro_MAE':float(p_table.direct.mean()),'gated_pure_head_macro_MAE':float(p_table['head'].mean())})
  pd.DataFrame(endpoint_scores).to_csv(O/'fast_pure_endpoint_accuracy.csv',index=False)
 dscore=metrics(test,pdirect);gscore=metrics(test,pgated)
 for meth,mm in [('direct_250',dscore),('OOF_endpoint_gated_150plus150',gscore)]:
  scores.append({'seed':seed,'model':meth,'test_n':len(test),'n_systems':len(set(system[test])),**{k:float(v) for k,v in mm.items()}})
 pd.DataFrame(scores).to_csv(O/'fast_comparison_metrics.csv',index=False)
 err=pd.DataFrame({'system':system[test],'direct':abs(y[test]-pdirect),'gated':abs(y[test]-pgated)}).groupby('system').mean()
 diff=(err.gated-err.direct).to_numpy();brng=np.random.default_rng(seed+1001)
 # Vectorized bootstrap 1000 x number of systems
 sampled=brng.integers(0,len(diff),(1000,len(diff)));bs=diff[sampled].mean(axis=1)
 errors.append({'seed':seed,'system_macro_delta_gated_minus_direct':float(diff.mean()),'CI95_low':float(np.quantile(bs,.025)),'CI95_high':float(np.quantile(bs,.975)),'n_systems':len(diff)})
 pd.DataFrame(errors).to_csv(O/'fast_system_bootstrap.csv',index=False)
 for label,mask in [('very_low',y[test]<-5),('middle',(y[test]>=-3)&(y[test]<-2)),('high',y[test]>-1)]:
  if mask.any():
   strata.append({'seed':seed,'segment':label,'test_n':int(mask.sum()),'direct_MAE':float(np.mean(abs(y[test][mask]-pdirect[mask]))),'gated_MAE':float(np.mean(abs(y[test][mask]-pgated[mask])))})
 pd.DataFrame(strata).to_csv(O/'fast_tail_strata.csv',index=False)
 # export only tiny aggregated results, do not redistribute labeled rowwise predictions
 print('END',seed,dscore,gscore,'elapsed',round(time.monotonic()-st,1),'seconds',flush=True)
meta={'input_raw_md5':hashlib.md5(raw.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256(a.metadata.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'package_versions':{'rdkit':rdBase.rdkitVersion,'lightgbm':lightgbm.__version__,'python':platform.python_version()},'note':'Exploratory post hoc computational amendment; threefold solute OOF pure-head for residual training; unequal architecture capacity; no test labels used in fit'}
(O/'fast_provenance.json').write_text(json.dumps(meta,indent=2))
print('ALL_DONE',flush=True)
