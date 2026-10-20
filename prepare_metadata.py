#!/usr/bin/env python3
"""Portable raw MixtureSolDB v3 -> canonical CHEM-02 metadata.
No test-set measurements are used during any model fitting here; preprocessing
canonicalizes structures/units and groups physically identical measurement conditions.
Processed derivative is intentionally LOCAL ONLY; do not redistribute without license review.
"""
from __future__ import annotations
import argparse, hashlib, re, json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
MD5='597e31685fe36c78d568f3d37b6e1336'
PEG_RE=re.compile(r'\bpeg\s*[-_]?\s*\d*\b|poly\s*\(?\s*ethylene\s*glycol|polyethylene\s*glycol',re.I)
@lru_cache(None)
def canon(x):
 if not isinstance(x,str) or not x.strip():return None
 m=Chem.MolFromSmiles(x.strip())
 return Chem.MolToSmiles(m,isomericSmiles=True) if m else None
@lru_cache(None)
def mw(x):
 return float(Descriptors.MolWt(Chem.MolFromSmiles(x))) if x is not None else float('nan')
@lru_cache(None)
def scaffold(x):
 m=Chem.MolFromSmiles(x)
 return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) or ('ACYCLIC:'+x)

def prepare(raw_path:Path,output:Path):
 if hashlib.md5(raw_path.read_bytes()).hexdigest()!=MD5:raise ValueError('Wrong MixtureSolDB v3 checksum')
 r=pd.read_csv(raw_path,low_memory=False)
 for c in ['Temperature_K','Fraction_Solvent1','Fraction_Solvent2','LogS(mole_fraction)']:
  r[c]=pd.to_numeric(r[c],errors='coerce')
 typ=r.Fraction_Type.fillna('').astype(str).str.strip().str.lower()
 m=typ.eq('mass').to_numpy(); q=typ.eq('mole').to_numpy()
 names=r.Solvent1.fillna('').astype(str)+' / '+r.Solvent2.fillna('').astype(str)
 peg=(names.str.contains(PEG_RE,regex=True)&typ.eq('mass')).to_numpy()
 sol=r.SMILES_Solute.map(canon);s1=r.SMILES_Solvent1.map(canon);s2=r.SMILES_Solvent2.map(canon)
 f1=r.Fraction_Solvent1.to_numpy(float); f2=r.Fraction_Solvent2.to_numpy(float)
 mw1=s1.map(mw).to_numpy(float); mw2=s2.map(mw).to_numpy(float)
 with np.errstate(all='ignore'):
  n1=np.where(m,f1/mw1,f1);n2=np.where(m,f2/mw2,f2)
  z1=n1/(n1+n2)
 x1=s1.fillna('').to_numpy(str); x2=s2.fillna('').to_numpy(str)
 flip=x1>x2;A=np.where(flip,x2,x1);B=np.where(flip,x1,x2)
 z=np.where(flip,1-z1,z1)
 temp=r.Temperature_K.to_numpy(float); y=r['LogS(mole_fraction)'].to_numpy(float)
 flags={'ambiguous_PEG_mass':peg,'invalid_smiles':(sol.isna()|s1.isna()|s2.isna()).to_numpy(),
  'invalid_fraction':~((m|q)&np.isfinite(z1)&(z1>=0)&(z1<=1)&np.isfinite(f1+f2)&(abs(f1+f2-1)<1e-4)&(f1>=0)&(f2>=0)),
  'invalid_log_solubility':~(np.isfinite(y)&(y<=0)),
  'invalid_temperature':~((temp>=180)&(temp<=500)),
  'same_solvent_pair':A==B}
 allowed=np.ones(len(r),dtype=bool)
 for flag in flags.values():allowed&=~flag
 ix=np.flatnonzero(allowed)
 good=pd.DataFrame({'solute':sol.to_numpy()[allowed],
  'pair':pd.Series(A[allowed])+' || '+pd.Series(B[allowed]),
  'T_key':np.round(temp[allowed],3),'z_key':np.round(z[allowed],8),'y':y[allowed],
  'source':r.Source.iloc[ix].astype('string').fillna('MISSING').str.strip().str.lower().to_numpy(),
  'T':temp[allowed],'z':z[allowed],
  'endpoint':np.isclose(z[allowed],0,atol=1e-9)|np.isclose(z[allowed],1,atol=1e-9)})
 keys=['solute','pair','T_key','z_key']
 grp=good.groupby(keys,dropna=False,sort=True,observed=True)
 d=grp.agg(y=('y','median'),n_replicates=('y','size'),source_count=('source','nunique'),T=('T','median'),z=('z','median'),endpoint=('endpoint','max')).reset_index()
 d=d.merge(grp['source'].first().reset_index(name='source'),on=keys,how='left',validate='one_to_one')
 d['A']=d.pair.str.split(' || ',regex=False).str[0]
 d['B']=d.pair.str.split(' || ',regex=False).str[1]
 d['system']=d.solute+' /// '+d.pair
 d['scaffold']=d.solute.map(scaffold)
 d['pure_solvent']=np.where(d.endpoint,np.where(d.z>.5,d.A,d.B),None)
 d['RecordKey']=d.solute+'|'+d.pair+'|'+d.T_key.astype(str)+'|'+d.z_key.astype(str)
 assert d.RecordKey.is_unique
 assert len(r)==175166 and len(d)==172880,(len(r),len(d))
 assert ((d.A+' || '+d.B)==d.pair).all()
 output.parent.mkdir(parents=True,exist_ok=True)
 d.to_csv(output,index=False)
 audit={'raw_md5':MD5,'raw_rows':len(r),'valid_before_dedup':int(allowed.sum()),'harmonized_rows':len(d),
  'duplicates_removed':int(allowed.sum()-len(d)),'excluded':int((~allowed).sum()),
  'exclusion_flags_raw_overlapping':{k:int(v.sum()) for k,v in flags.items()},
  'solutes':d.solute.nunique(),'solvent_pairs':d.pair.nunique(),'systems':d.system.nunique(),
  'processed_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
 output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps(audit,indent=2))
 return d
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();prepare(a.raw,a.out)
