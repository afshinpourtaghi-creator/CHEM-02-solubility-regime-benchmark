#!/usr/bin/env python3
"""Phase 16: Classical binary-solvent CNIBS/Redlich-Kister / Jouyban-Acree-type comparisons.
Reads verified ORIGINAL CSV and existing Phase4 harmonized derivative.
Crucial: same-system endpoints are OBSERVED (oracle); this is NOT zero-shot prediction.
Global J coefficients may be trained only on different LITERATURE SOURCES.
Local J coefficients use 3 observed interior composition points of SAME curve.
This script deliberately does not compare these tasks as equals to zero-shot ML.
"""
from __future__ import annotations
import hashlib, json, os, platform, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

ap=argparse.ArgumentParser()
ap.add_argument('--raw',required=True,type=Path)
ap.add_argument('--metadata',required=True,type=Path)
ap.add_argument('--out',required=True,type=Path)
a=ap.parse_args()
RAW=a.raw; META=a.metadata; OUT=a.out; OUT.mkdir(exist_ok=True,parents=True)
EXPECTED='597e31685fe36c78d568f3d37b6e1336'
assert hashlib.md5(RAW.read_bytes()).hexdigest()==EXPECTED
raw=pd.read_csv(RAW)
assert len(raw)==175166 and len(raw.columns)==19
meta=pd.read_csv(META)
assert len(meta)==172880 and meta.RecordKey.is_unique
raw_unit=raw.Fraction_Type.astype(str).str.lower().str.strip().value_counts().to_dict()
fr_sum=raw['Fraction_Solvent1'].add(raw['Fraction_Solvent2'])
raw_y=np.log10(pd.to_numeric(raw['Solubility(mole_fraction)'],errors='coerce'))
reported=pd.to_numeric(raw['LogS(mole_fraction)'],errors='coerce')
raw_audit={'raw_md5':EXPECTED,'raw_rows':len(raw),'raw_columns':len(raw.columns),'fraction_types':raw_unit,
           'fraction_sum_gt_1e_minus_4':int((abs(fr_sum-1)>1e-4).sum()),
           'log_solubility_disagreement_gt_1e_minus_3':int((abs(raw_y-reported)>1e-3).sum()),
           'harmonized_records':len(meta),'harmonized_sha256':hashlib.sha256(META.read_bytes()).hexdigest(),
           'caveat':'Reuses Phase4 harmonization; raw-to-clean transforms not independently reconstructed in Phase16.'}
(OUT/'raw_integrity_audit.json').write_text(json.dumps(raw_audit,indent=2,ensure_ascii=False))
print('AUDIT',raw_audit,flush=True)
# Filter single-source curves. Endpoint = z=0 or 1 and pure_endpoint flag; include >=5 distinct interior.
d=meta[(meta.source_count==1)&meta.y.notna()&meta.z.between(0,1)&meta['T'].notna()].copy()
d['source']=d.source.fillna('UNKNOWN').astype(str)
# grouping key includes T and source; prevents mixing conditions from different publications
keys=['solute','pair','T_key','source']
curves=[];points=[]
# Each curve is strictly sorted composition; no failed endpoints silently substituted from unrelated sources.
for key,g in d.groupby(keys,sort=False,dropna=False):
    g=g.sort_values('z')
    zz=g.z.to_numpy(float); yy=g.y.to_numpy(float)
    ep_a=np.where(np.isclose(zz,1.,atol=1e-8))[0];ep_b=np.where(np.isclose(zz,0.,atol=1e-8))[0]
    if len(ep_a)!=1 or len(ep_b)!=1:continue
    if not (bool(g.endpoint.iloc[ep_a[0]]) and bool(g.endpoint.iloc[ep_b[0]])):continue
    interiors=np.flatnonzero((zz>1e-8)&(zz<1-1e-8))
    if len(interiors)<5:continue
    if np.unique(zz[interiors]).size!=len(interiors):continue
    T=float(g['T'].iloc[0]);
    cid=len(curves); yA=float(yy[ep_a[0]]);yB=float(yy[ep_b[0]])
    curves.append({'curve_id':cid,'source':str(key[3]),'solute':str(key[0]),'pair':str(key[1]),'T':T,
                   'system':str(key[0])+' /// '+str(key[1]),'n_interior':len(interiors),
                   'record_key_A':g.RecordKey.iloc[ep_a[0]],'record_key_B':g.RecordKey.iloc[ep_b[0]]})
    # indexed holdout: assign nearest 0.2,0.5,0.8 without replacements, deterministic tie-break sorted z
    ii=interiors.tolist(); selected=[]
    for target in (.2,.5,.8):
        rem=[j for j in ii if j not in selected]
        pick=min(rem,key=lambda j:(abs(zz[j]-target),zz[j]))
        selected.append(pick)
    for j in interiors:
        z=float(zz[j]);
        points.append({'curve_id':cid,'source':str(key[3]),'system':str(key[0])+' /// '+str(key[1]),
                       'record_key':g.RecordKey.iloc[j],'T':T,'z':z,'y':float(yy[j]),
                       'yA':yA,'yB':yB,'is_local_calibration':j in selected})
cv=pd.DataFrame(curves);pt=pd.DataFrame(points)
assert len(pt)>10000 and len(cv)>1000
# test compositions are withheld from each local fitting; endpoints are always known.
pt['ideal']=pt.z*pt.yA+(1-pt.z)*pt.yB
pt['resid']=pt.y-pt.ideal
pt['phi0']=(pt.z*(1-pt.z)*298.15/pt["T"])
pt['phi1']=pt.phi0*(2*pt.z-1)
pt['phi2']=pt.phi1*(2*pt.z-1)
# LOCAL restricted-data models: one- to three-parameter classical polynomial in 2z-1.
local_res=[]
for cid,g in pt.groupby('curve_id',sort=False):
    cc=g[g.is_local_calibration];tt=g[~g.is_local_calibration]
    assert len(cc)==3 and len(tt)>=2
    row={'curve_id':cid,'source':g.source.iloc[0],'system':g.system.iloc[0],'T':float(g.T.iloc[0]) if False else float(g['T'].iloc[0]),
         'n_local_test':len(tt),'n_local_calibration':3}
    row['ideal_curve_MAE']=float(np.abs(tt.y-tt.ideal).mean())
    for n in (1,2,3):
        cols=['phi0','phi1','phi2'][:n]
        X=cc[cols].to_numpy(float); target=cc.resid.to_numpy(float)
        coef=np.linalg.lstsq(X,target,rcond=None)[0]
        pred=tt.ideal.to_numpy(float)+tt[cols].to_numpy(float)@coef
        # No artificially clipped target; do not create physically invalid solubility >1 via clipping.
        row[f'JA_{n}term_curve_MAE']=float(np.mean(np.abs(tt.y.to_numpy(float)-pred)))
        row[f'JA_{n}term_curve_RMSE']=float(np.sqrt(np.mean((tt.y.to_numpy(float)-pred)**2)))
        row[f'JA_{n}term_unphysical_gt_zero_fraction']=float(np.mean(pred>0))
        row[f'JA_{n}term_max_abs_coefficient']=float(np.max(np.abs(coef)))
    local_res.append(row)
local=pd.DataFrame(local_res);local.to_csv(OUT/'local_curve_threepoint_metrics.csv',index=False)
# SOURCE-DISJOINT globally shared correction model, with *measured endpoints of TEST sources*, explicitly not zero-shot.
# source-level split never lets model see source-specific interior measurements; direct endpoints on test provided.
allsource=np.array(sorted(cv.source.unique()))
source_rows=[];global_pred_records=[];global_coefficients=[]
for seed in (1601,1602,1603):
    rng=np.random.default_rng(seed)
    held=set(rng.choice(allsource,size=int(np.ceil(.20*len(allsource))),replace=False).tolist())
    tr=pt[~pt.source.isin(held)];te=pt[pt.source.isin(held)]
    assert set(tr.source).isdisjoint(set(te.source))
    # equal total weight across curves (rather than all rows dominated by long curves)
    cnt=tr.groupby('curve_id').curve_id.transform('size').to_numpy(float)
    w=1/cnt
    for n in (0,1,2,3):
        cols=['phi0','phi1','phi2'][:n]
        if n==0:
            pred=te.ideal.to_numpy(float);coef=[]
        else:
            X=tr[cols].to_numpy(float);Y=tr.resid.to_numpy(float)
            Xw=X*np.sqrt(w)[:,None];Yw=Y*np.sqrt(w)
            # small a-priori numerical penalty stabilizes weakly identified high-order term
            coef=np.linalg.solve(Xw.T@Xw+1e-3*np.eye(n),Xw.T@Yw)
            pred=te.ideal.to_numpy(float)+te[cols].to_numpy(float)@coef
        test=te[['curve_id','system','source','record_key','y']].copy()
        test['prediction']=pred;test['abs_error']=np.abs(test.y-test.prediction)
        er=test.groupby('system').abs_error.mean()
        cr=test.groupby('curve_id').abs_error.mean()
        source_rows.append({'seed':seed,'n_correction_terms':n,'n_fit_sources':len(set(tr.source)),'n_test_sources':len(set(te.source)),
                            'n_test_interior':len(te),'n_test_curves':len(set(te.curve_id)),'n_test_systems':len(er),
                            'micro_MAE':float(test.abs_error.mean()),'curve_macro_MAE':float(cr.mean()),
                            'system_macro_MAE':float(er.mean())})
        global_coefficients.append({'seed':seed,'n_terms':n,'k0':float(coef[0]) if n>=1 else np.nan,
                                     'k1':float(coef[1]) if n>=2 else np.nan,'k2':float(coef[2]) if n>=3 else np.nan})
        # error across same test sets for paired differences; summarize per system
        tmp=er.rename('mae').reset_index();tmp['seed']=seed;tmp['n_terms']=n
        global_pred_records.append(tmp)
        print('GLOBAL',seed,n,'systems',len(er),'macro',round(er.mean(),4),flush=True)
G=pd.DataFrame(source_rows);G.to_csv(OUT/'global_source_disjoint_metrics.csv',index=False)
pd.DataFrame(global_coefficients).to_csv(OUT/'global_coefficients.csv',index=False)
GP=pd.concat(global_pred_records,ignore_index=True)
# Paired bootstrap across systems test data; within seed, not independent across seeds
ci=[]
for seed,g in GP.groupby('seed'):
    piv=g.pivot(index='system',columns='n_terms',values='mae')
    rr=np.random.default_rng(seed+14123)
    for method in (1,2,3):
        diff=(piv[method]-piv[0]).dropna().to_numpy(float)
        ix=rr.integers(0,len(diff),size=(2000,len(diff)))
        sims=diff[ix].mean(axis=1)
        ci.append({'seed':seed,'contrast':f'JA_{method}term_minus_ideal','n_systems':len(diff),
                   'delta_system_macro_MAE':float(diff.mean()),'CI_95_low':float(np.quantile(sims,.025)),
                   'CI_95_high':float(np.quantile(sims,.975))})
pd.DataFrame(ci).to_csv(OUT/'paired_system_bootstrap.csv',index=False)
# Local 3-point per-curve metrics, aggregate equal weights per system and per curve; these are correlated.
local_melt=local.melt(id_vars=['curve_id','system','source','n_local_test'],value_vars=['ideal_curve_MAE','JA_1term_curve_MAE','JA_2term_curve_MAE','JA_3term_curve_MAE'],var_name='method',value_name='curve_mae')
local_overall=local_melt.groupby('method',as_index=False).agg(curve_macro_MAE=('curve_mae','mean'),median_curve_MAE=('curve_mae','median'))
sys=local_melt.groupby(['method','system'],as_index=False).curve_mae.mean()
local_overall=local_overall.merge(sys.groupby('method',as_index=False).curve_mae.mean().rename(columns={'curve_mae':'system_macro_MAE'}),on='method')
local_overall.to_csv(OUT/'local_threepoint_summary.csv',index=False)
# Paired CLUSTER bootstrap at unique solute/solvent-pair system level.
local_ci=[]
by_system=local.groupby('system',sort=False)[['ideal_curve_MAE','JA_1term_curve_MAE','JA_2term_curve_MAE','JA_3term_curve_MAE']].mean()
rg=np.random.default_rng(160160)
for ref in ['ideal_curve_MAE','JA_2term_curve_MAE']:
    v=(by_system['JA_3term_curve_MAE']-by_system[ref]).to_numpy(float)
    sim=v[rg.integers(0,len(v),(2000,len(v)))].mean(axis=1)
    local_ci.append({'contrast':'JA_3term_minus_'+ref,'n_systems':len(v),'delta_system_macro':float(v.mean()),'CI_95_low':float(np.quantile(sim,.025)),'CI_95_high':float(np.quantile(sim,.975))})
pd.DataFrame(local_ci).to_csv(OUT/'local_paired_system_bootstrap.csv',index=False)

# Robustness by number of points, helps assess data availability
local['interior_points']=local.n_local_test+3
local['bin_n']=pd.cut(local.interior_points,bins=[4,7,10,15,1000],labels=['5–7','8–10','11–15','16+'])
local.groupby('bin_n',observed=True)[['ideal_curve_MAE','JA_1term_curve_MAE','JA_2term_curve_MAE','JA_3term_curve_MAE']].mean().reset_index().to_csv(OUT/'local_strata_npoints.csv',index=False)
# metadata only export, not record-level user data
raw_audit.update({'qualified_curves':len(cv),'qualified_interior_points':len(pt),'qualified_systems':int(cv.system.nunique()),
                  'qualified_sources':int(cv.source.nunique()),'local_holdout_interior_points':int((~pt.is_local_calibration).sum()),
                  'median_interior_per_curve':float(cv.n_interior.median()),
                  'local_3term_mean_unphysical_gt_zero_fraction':float(local['JA_3term_unphysical_gt_zero_fraction'].mean()),
                  'local_3term_curves_with_invalid_extrapolation':int((local['JA_3term_unphysical_gt_zero_fraction']>0).sum()),'method':'Jouyban-Acree/CNIBS-RK-style composition expansion, adapted to MOLE-fraction solvent composition; conditional on SAME-CURVE experimental endpoints. Local model additionally uses 3 mixed-solvent measurements of test curve. Global model derives coefficients only from training sources, but test endpoints are observed.'})
(OUT/'raw_integrity_audit.json').write_text(json.dumps(raw_audit,indent=2,ensure_ascii=False))
print('LOCAL SUMMARY\n',local_overall.to_string(index=False),flush=True)
print('AUDIT_CURVES',len(cv),len(pt),flush=True)
