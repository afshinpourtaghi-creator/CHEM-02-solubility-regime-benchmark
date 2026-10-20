#!/usr/bin/env python3
"""Independent reconstruction of published CHEM-02 harmonization (v3 raw -> audit only).

This implementation does NOT import the historical preprocessing code, and does not
export transformed row-level raw records. It reports aggregate diagnostics and
comparison to a separately stored historical work-product.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

PEG_RE = re.compile(r'\bpeg\s*[-_]?\s*\d*\b|poly\s*\(?\s*ethylene\s*glycol|polyethylene\s*glycol',re.I)
@lru_cache(None)
def normalize_smiles(s):
    if not isinstance(s,str) or not s.strip():return None
    mol=Chem.MolFromSmiles(s.strip())
    return Chem.MolToSmiles(mol,isomericSmiles=True) if mol is not None else None
@lru_cache(None)
def mol_mass(s):
    if s is None:return float('nan')
    return float(Descriptors.MolWt(Chem.MolFromSmiles(s)))

def audit(rawfile,historical,out):
    out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.md5(rawfile.read_bytes()).hexdigest()
    assert digest=='597e31685fe36c78d568f3d37b6e1336',digest
    raw=pd.read_csv(rawfile,low_memory=False)
    for col in ['Temperature_K','Fraction_Solvent1','Fraction_Solvent2','LogS(mole_fraction)','Solubility(mole_fraction)']:
        raw[col]=pd.to_numeric(raw[col],errors='coerce')
    typ=raw.Fraction_Type.fillna('').astype(str).str.strip().str.lower()
    m=typ.eq('mass').to_numpy(); q=typ.eq('mole').to_numpy()
    names=raw.Solvent1.fillna('').astype(str)+' / '+raw.Solvent2.fillna('').astype(str)
    peg=(names.str.contains(PEG_RE,regex=True)&typ.eq('mass')).to_numpy()
    sol=raw.SMILES_Solute.map(normalize_smiles)
    s1=raw.SMILES_Solvent1.map(normalize_smiles)
    s2=raw.SMILES_Solvent2.map(normalize_smiles)
    mw1=s1.map(mol_mass).to_numpy(float);mw2=s2.map(mol_mass).to_numpy(float)
    f1=raw.Fraction_Solvent1.to_numpy(float);f2=raw.Fraction_Solvent2.to_numpy(float)
    # Avoid NaN/undefined on malformed source rows; these are audited below.
    with np.errstate(all='ignore'):
        n1=np.where(m,f1/mw1,f1);n2=np.where(m,f2/mw2,f2)
        z1=n1/(n1+n2)
    x1=s1.fillna('').to_numpy(str);x2=s2.fillna('').to_numpy(str)
    flip=x1>x2
    A=np.where(flip,x2,x1);B=np.where(flip,x1,x2)
    z=np.where(flip,1-z1,z1)
    y=raw['LogS(mole_fraction)'].to_numpy(float)
    temp=raw.Temperature_K.to_numpy(float)
    flags={
      'peg_mass_ambiguous':peg,
      'invalid_smiles':(sol.isna()|s1.isna()|s2.isna()).to_numpy(),
      'invalid_fraction':~((m|q)&np.isfinite(z1)&(z1>=0)&(z1<=1)&np.isfinite(f1+f2)&(abs(f1+f2-1)<1e-4)&(f1>=0)&(f2>=0)),
      'invalid_log_solubility':~(np.isfinite(y)&(y<=0)),
      'invalid_temperature':~((temp>=180)&(temp<=500)),
      'same_solvent_pair':(A==B),
    }
    allowed=np.ones(len(raw),bool)
    for val in flags.values():allowed &= ~val
    good=pd.DataFrame({'solute':sol.to_numpy()[allowed], 'pair':pd.Series(A[allowed])+' || '+pd.Series(B[allowed]),
      'T_key':np.round(temp[allowed],3),'z_key':np.round(z[allowed],8),'y':y[allowed],
      'source':raw.Source.iloc[np.flatnonzero(allowed)].astype('string').fillna('MISSING').str.strip().str.lower().to_numpy(),
      'T':temp[allowed],'z':z[allowed],
      'endpoint':np.isclose(z[allowed],0.,atol=1e-9)|np.isclose(z[allowed],1.,atol=1e-9)})
    keys=['solute','pair','T_key','z_key']
    grouped=good.groupby(keys,dropna=False,sort=True,observed=True)
    new=grouped.agg(y=('y','median'),n_replicates=('y','size'),source_count=('source','nunique'),
        T=('T','median'),z=('z','median'),endpoint=('endpoint','max')).reset_index()
    # historical representative source depends on input row order; preserve first within the raw grouping.
    first=grouped['source'].first().reset_index(name='source')
    new=new.merge(first,on=keys,how='left',validate='one_to_one')
    old=pd.read_csv(historical,low_memory=False)
    merged=old.merge(new,on=keys,how='outer',indicator=True,validate='one_to_one',suffixes=('_historical','_rebuild'))
    match={}
    for col in ['y','z','T']:
        lhs=merged[f'{col}_historical'].to_numpy(float);rhs=merged[f'{col}_rebuild'].to_numpy(float)
        valid=np.isfinite(lhs)&np.isfinite(rhs)
        match[col]={'count_mismatch_above_1e-10':int(np.sum(valid&(abs(lhs-rhs)>1e-10))),
                    'max_abs_delta':float(np.max(abs(lhs[valid]-rhs[valid]))) if valid.any() else None}
    for col in ['n_replicates','source_count','endpoint','source']:
        l=merged[f'{col}_historical'];r=merged[f'{col}_rebuild']
        match[col]={'mismatched':int(((l!=r)&(l.notna())&(r.notna())).sum())}
    dd=good.groupby(keys).agg(n=('y','size'),span=('y',lambda t: t.max()-t.min()),n_sources=('source','nunique'))
    rep=dd[dd.n>1]
    manual_flags=pd.DataFrame([{'reason':k,'count':int(v.sum())} for k,v in flags.items()])
    manual_flags.to_csv(out/'exclusion_reason_counts.csv',index=False)
    yexp=raw['Solubility(mole_fraction)'].to_numpy(float)
    with np.errstate(all='ignore'): yy=np.log10(yexp)
    delta=np.abs(yy-y)
    src_missing=int(raw.Source.isna().sum())
    original_endpoint=raw.IsPureSolventEndpoint.astype(str).str.lower().isin(['true','1','yes','y']).to_numpy()
    endpoint_computed=np.isclose(f1,0,atol=1e-9)|np.isclose(f1,1,atol=1e-9)
    # Use ORIGINAL reported composition to audit endpoint flags independently
    # of SMILES or molar-mass parsing (which can fail for excluded raw records).
    stats={
      'input_file_name':rawfile.name,'input_md5':digest,'rdkit_version':rdBase.rdkitVersion,
      'raw_rows':len(raw),'raw_columns':len(raw.columns),
      'fraction_types':typ.value_counts(dropna=False).to_dict(),
      'ambiguous_PEG_mass_excluded':int(peg.sum()),
      'excluded_other_unique':int((~allowed&~peg).sum()),
      'excluded_total_unique':int((~allowed).sum()),
      'valid_before_dedup':int(allowed.sum()),
      'rebuild_unique_physical_conditions':int(len(new)),
      'historical_unique_physical_conditions':int(len(old)),
      'comparison':{'keys_in_both':int((merged._merge=='both').sum()),
                    'only_in_historical':int((merged._merge=='left_only').sum()),
                    'only_in_rebuild':int((merged._merge=='right_only').sum()),'column_diffs':match},
      'repeat_groups':int(len(rep)),'repeat_extra_rows':int((rep.n-1).sum()),
      'repeat_groups_gt_0p1':int((rep.span>0.1).sum()),
      'repeat_groups_cross_source':int((rep.n_sources>1).sum()),
      'rowwise_logS_disagreement_gt_1e-3':int(np.sum(np.isfinite(delta)&(delta>1e-3))),
      'source_missing_count':src_missing,
      'endpoint_flag_mismatch_raw':int((original_endpoint!=endpoint_computed).sum()),
      'unique_solutes':int(new.solute.nunique()),'unique_solvent_pairs':int(new.pair.nunique()),
      'unique_systems':int(new[['solute','pair']].drop_duplicates().shape[0]),
      'unique_sources_before_collapse':int(good.source.nunique()),
      'multi_source_collapsed_groups':int((new.source_count>1).sum()),
      'cleaning_rules':{'temperature_K':[180,500], 'solubility_log10_max':0,
        'fraction_sum_tol':1e-4,'group_T_round_decimals':3,'group_z_round_decimals':8,
        'group_output':'median measured logS within same canonical solute/solvent pair/T/z',
        'PEG_detection':'conservative name-based PEG screen on mass fraction records; not comprehensive'}
    }
    (out/'independent_rebuild_audit.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False,default=int))
    return stats
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--historical',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(audit(args.raw,args.historical,args.out),indent=2,ensure_ascii=False,default=int))
