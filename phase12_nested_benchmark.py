#!/usr/bin/env python3
"""CHEM-02 phase 12; locked, test-label-isolated nested group holdout.
Usage: python phase12_nested_benchmark.py --seeds 901,902 --splits source_disjoint,joint_chemical_OOD
Requires MixtureSolDB.csv and phase-4 harmonized_metadata.csv in /mnt/data.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='2'
import argparse,hashlib,json,platform,time
from pathlib import Path
import pandas as pd,numpy as np
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from lightgbm import LGBMRegressor
import lightgbm
RDLogger.DisableLog('rdApp.*')
a=argparse.ArgumentParser();a.add_argument('--input',default='/mnt/data/CHEM02_REAL_RESULTS/harmonized_metadata.csv');a.add_argument('--raw',default='/mnt/data/MixtureSolDB.csv');a.add_argument('--out',default='/mnt/data/CHEM02_PHASE12');a.add_argument('--seeds',default='901,902');a.add_argument('--splits',default='source_disjoint,joint_chemical_OOD');a.add_argument('--threads',type=int,default=3); args=a.parse_args()
out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
md5=hashlib.md5(Path(args.raw).read_bytes()).hexdigest();assert md5=='597e31685fe36c78d568f3d37b6e1336',md5
D=Path(args.input);d=pd.read_csv(D);assert len(d)==172880 and d.RecordKey.is_unique
assert d.y.le(0).all() and d.z.between(0,1).all()
# A/B must align to the canonical pair and canonical molar composition z.
assert ((d.A+' || '+d.B)==d.pair).all(), 'Canonical solvent ordering mismatch'
N=len(d);valid=d.source_count.to_numpy()==1;interior=~d.endpoint.to_numpy(bool)
Y=d.y.to_numpy(np.float64);Z=d.z.to_numpy(np.float32);T=d['T'].to_numpy(np.float32)
SYS=d.system.to_numpy();sol=d.solute.to_numpy();pair=d.pair.to_numpy();sour=d.source.to_numpy()
sub_groups={key:pd.factorize(d[key],sort=True)[0] for key in ('source','solute','pair')}
all_ix=np.arange(N)
print('AUDIT',N,int(valid.sum()),'chemical entities',len(set(sol)|set(d.A)|set(d.B)),flush=True)

generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=128)
smiles=sorted(set(sol)|set(d.A)|set(d.B));emb={}
for s in smiles:
 m=Chem.MolFromSmiles(s)
 if m is None: raise ValueError(s)
 fp=np.zeros(128,np.float32);DataStructs.ConvertToNumpyArray(generator.GetFingerprint(m),fp)
 desc=np.array([Descriptors.MolWt(m)/500,Descriptors.MolLogP(m)/10,Descriptors.TPSA(m)/200,Descriptors.NumHDonors(m)/10,Descriptors.NumHAcceptors(m)/10],np.float32)
 emb[s]=np.r_[fp,desc]
def get_emb(x):return np.stack([emb[q] for q in x]).astype(np.float32)
S=get_emb(sol);A=get_emb(d.A.to_numpy());B=get_emb(d.B.to_numpy());M=Z[:,None]*A+(1-Z)[:,None]*B
G=Z*(1-Z);U=(1000/T-1000/298.15).astype(np.float32)
Xbasic=np.column_stack((S,M,np.abs(A-B),S[:,:128]*M[:,:128],Z,G,T/300,300/T)).astype(np.float32)
# Strictly pre-prediction information. Molecular descriptors: 128+ [MW, logP, TPSA, HBD, HBA].
sa=S[:,128:];ma=M[:,128:];ab=np.abs(A[:,128:]-B[:,128:])
extras=np.column_stack((U,U**2,G*U,Z*U,(1-Z)*U,sa[:,1]-ma[:,1],(sa[:,1]-ma[:,1])**2,ab[:,1],G*ab[:,1],G*ab[:,2],ma[:,3]*sa[:,4],ma[:,4]*sa[:,3],G*(ma[:,3]*sa[:,4]+ma[:,4]*sa[:,3]),U*ma[:,1],U*ma[:,2],U*sa[:,1])).astype(np.float32)
Xphys=np.column_stack((Xbasic,extras)).astype(np.float32)
features={'direct_standard':Xbasic,'physics_features':Xphys}
del S,A,B,M,sa,ma,ab

candidates=[(15,2.),(31,4.),(63,8.)];rounds=[150,350,650]

def source_split(base,seed,fraction=.20):
 rng=np.random.default_rng(seed); src=sub_groups['source'];uniq=np.unique(src[base]);choose=rng.choice(uniq,size=max(1,int(len(uniq)*fraction)),replace=False)
 hold=np.isin(src,choose);return base[~hold[base]],base[hold[base]&interior[base]]

def chemical_split(base,seed,fraction=.22):
 rng=np.random.default_rng(seed);g=sub_groups['solute'];h=sub_groups['pair'];us=np.unique(g[base]);up=np.unique(h[base]);
 gs=rng.choice(us,size=max(1,int(len(us)*fraction)),replace=False);hp=rng.choice(up,size=max(1,int(len(up)*fraction)),replace=False)
 sg=np.isin(g,gs);pg=np.isin(h,hp)
 return base[~(sg[base]|pg[base])],base[sg[base]&pg[base]&interior[base]]

def split(base,seed,kind):
 if kind=='source_disjoint':return source_split(base,seed)
 if kind=='joint_chemical_OOD':return chemical_split(base,seed)
 raise ValueError(kind)

def macro(err,sys):return float(pd.DataFrame({'sys':sys,'e':np.abs(err)}).groupby('sys',sort=False).e.mean().mean())
def model(seed,leaves,regularization,trees):
 return LGBMRegressor(n_estimators=trees,learning_rate=.06,num_leaves=leaves,reg_lambda=regularization,
            min_child_samples=35,colsample_bytree=.85,subsample=.85,verbosity=-1,random_state=seed,n_jobs=args.threads)

rows=[];scores=[];audits=[];cirows=[]
for kind in args.splits.split(','):
 for seed in [int(x) for x in args.seeds.split(',')]:
  tic=time.monotonic();outer_fit,outer_test=split(all_ix[valid],seed,kind);inner_fit,inner_val=split(outer_fit,seed+8713,kind)
  print('SPLIT',kind,seed,'outer',len(outer_fit),len(outer_test),'inner',len(inner_fit),len(inner_val),flush=True)
  assert len(outer_test)>500 and len(inner_val)>500 and len(inner_fit)>30000
  assert not (set(outer_fit)&set(outer_test) or set(inner_fit)&set(inner_val))
  assert set(inner_fit).issubset(set(outer_fit)) and set(inner_val).issubset(set(outer_fit))
  if kind=='source_disjoint':
   assert not (set(sour[outer_fit])&set(sour[outer_test]));assert not (set(sour[inner_fit])&set(sour[inner_val]))
  else:
   for g in (sol,pair):
    assert not (set(g[outer_fit])&set(g[outer_test])); assert not (set(g[inner_fit])&set(g[inner_val]))
  audit={'split':kind,'seed':seed,'n_outer_fit':len(outer_fit),'n_outer_test':len(outer_test),
   'n_inner_fit':len(inner_fit),'n_inner_val':len(inner_val),'n_test_systems':len(set(SYS[outer_test])),
   'outer_shared_solutes':len(set(sol[outer_fit])&set(sol[outer_test])),
   'outer_shared_pairs':len(set(pair[outer_fit])&set(pair[outer_test])),
   'outer_shared_sources':len(set(sour[outer_fit])&set(sour[outer_test]))}
  audits.append(audit);pd.DataFrame(audits).to_csv(out/'split_audit.csv',index=False)
  preds={}
  for rep,X in features.items():
   best=None
   for leaves,reg in candidates:
    m=model(seed+79,leaves,reg,650);m.fit(X[inner_fit],Y[inner_fit]);
    for n in rounds:
     yhat=np.minimum(m.predict(X[inner_val],num_iteration=n),0)
     val=macro(Y[inner_val]-yhat,SYS[inner_val]);sc={'split':kind,'seed':seed,'representation':rep,'num_leaves':leaves,'reg_lambda':reg,'n_estimators':n,'inner_macro_MAE':val};scores.append(sc)
     print('INNER',kind,seed,rep,'leaves',leaves,'reg',reg,'trees',n,'macro',round(val,5),flush=True)
     if best is None or (val,n,leaves)<(best[0],best[1]['n_estimators'],best[1]['num_leaves']): best=(val,sc)
    pd.DataFrame(scores).to_csv(out/'inner_search.csv',index=False)
   sc=best[1];chosen=model(seed+79,sc['num_leaves'],sc['reg_lambda'],sc['n_estimators']);chosen.fit(X[outer_fit],Y[outer_fit]);yp=np.minimum(chosen.predict(X[outer_test]),0);preds[rep]=yp
   e=Y[outer_test]-yp
   rows.append({**audit,'representation':rep,'inner_best_macro_MAE':best[0],'selected_leaves':sc['num_leaves'],'selected_l2':sc['reg_lambda'],'selected_trees':sc['n_estimators'],
       'test_system_macro_MAE':macro(e,SYS[outer_test]),'test_micro_MAE':float(np.abs(e).mean()),'test_RMSE':float(np.sqrt(np.mean(e*e))), 'signed_bias':float(np.mean(e))})
   pd.DataFrame(rows).to_csv(out/'outer_metrics.csv',index=False)
   print('OUTER',kind,seed,rep,'macro',round(rows[-1]['test_system_macro_MAE'],5),'selected',sc['num_leaves'],sc['n_estimators'],flush=True)
  # paired system bootstrap; same sample and cluster set for two representations
  dif=pd.DataFrame({'system':SYS[outer_test],'diff':np.abs(Y[outer_test]-preds['physics_features'])-np.abs(Y[outer_test]-preds['direct_standard'])}).groupby('system',sort=False)['diff'].mean().to_numpy()
  rng=np.random.default_rng(seed+33017);boots=np.array([dif[rng.integers(0,len(dif),size=len(dif))].mean() for _ in range(1000)])
  cirows.append({'split':kind,'seed':seed,'delta_physics_minus_direct':float(dif.mean()),'ci95_low':float(np.quantile(boots,.025)),'ci95_high':float(np.quantile(boots,.975)),'n_test_systems':len(dif)})
  pd.DataFrame(cirows).to_csv(out/'paired_cluster_ci.csv',index=False)
  pd.DataFrame({'RecordKey':d.RecordKey.iloc[outer_test].to_numpy(),'system':SYS[outer_test],'y':Y[outer_test],**preds}).to_csv(out/f'_LOCAL_ONLY_predictions_{kind}_{seed}.csv.gz',index=False,compression='gzip')
  print('DONE',kind,seed,'elapsed',round(time.monotonic()-tic,1),'sec',flush=True)
prov={'file_md5':md5,'metadata_sha256':hashlib.sha256(D.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'protocol_sha256':hashlib.sha256((out/'LOCKED_PROTOCOL.md').read_bytes()).hexdigest(),'software':{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'lightgbm':lightgbm.__version__,'numpy':np.__version__,'pandas':pd.__version__},
 'warning':'Retrospective same-dataset nested holdout, not a genuinely untouched prospective test; physics feature enhancement does not enforce thermodynamic laws.'}
(out/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
print('FINISHED',len(rows),'outer model evaluations',flush=True)
