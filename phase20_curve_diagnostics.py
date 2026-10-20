#!/usr/bin/env python3
"""CHEM-02 phase20: threshold/shape audit of phase18 frozen predictions.
No experimental Supplementary Table S3 is imported. Never refits a model.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLDS=(0.,.01,.02,.05,.10,.15)
SOURCE_DOI='10.3390/molecules31071117'
def main():
  ap=argparse.ArgumentParser()
  ap.add_argument('--frozen',default='/mnt/data/CHEM02_PHASE18/FROZEN_external_grid_predictions.csv')
  ap.add_argument('--out',default='/mnt/data/CHEM02_PHASE20')
  args=ap.parse_args()
  src=Path(args.frozen); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
  d=pd.read_csv(src)
  assert len(d)==110, 'Unexpected number of frozen grid predictions'
  assert d.source_log10_mole_fraction.isna().all(), 'Unexpected external labels in the frozen file'
  assert d.study_doi.eq(SOURCE_DOI).all(), 'Mixed DOI in frozen predictions'
  assert set(d.solute_name)=={'hesperetin','trans-resveratrol'}
  assert len(d.groupby(['solute_name','solvent1_name','solvent2_name']))==10
  assert d[['solute_name','solvent1_name','solvent2_name','x_solvent1']].duplicated().sum()==0
  individual=[]
  for (solute,s1,s2),g in d.groupby(['solute_name','solvent1_name','solvent2_name'],sort=True):
    g=g.sort_values('x_solvent1');z=g.x_solvent1.to_numpy(float);yp=g.frozen_log10_mole_fraction_prediction.to_numpy(float)
    assert np.allclose(z,np.linspace(0,1,11),atol=1e-9)
    assert np.all(np.isfinite(yp))
    idx=int(np.argmax(yp));delta=float(yp[idx]-max(yp[0],yp[-1]));first=np.diff(yp)
    signs=np.sign(first);signs=signs[signs!=0]
    nturn=int(np.sum(signs[1:]!=signs[:-1]));total_var=float(np.sum(np.abs(first)))
    individual.append(dict(solute=solute,solvent1=s1,solvent2=s2,category='alcohol-ketone' if s2 in ('MEK','MIBK') else 'methanol-ethanol',
      frozen_peak_x_solvent2=float(1-z[idx]),logS_peak_margin_vs_best_endpoint=delta,
      endpoint_min_logS=float(min(yp[0],yp[-1])),endpoint_max_logS=float(max(yp[0],yp[-1])),
      grid_direction_changes=nturn,grid_total_variation_logS=total_var,
      grid_total_variation_to_range=float(total_var/(np.ptp(yp))) if np.ptp(yp) else np.nan,
      max_single_grid_step_logS=float(np.max(np.abs(first))),
      test_is_posthoc_and_unlabeled=True))
  bycurve=pd.DataFrame(individual)
  bycurve.to_csv(out/'frozen_curve_shape_diagnostics.csv',index=False)
  rows=[]
  for thr in THRESHOLDS:
    positive=bycurve.logS_peak_margin_vs_best_endpoint>thr+1e-12
    for category,g in bycurve.groupby('category'):
      group_pred=positive[bycurve.category==category]
      rows.append(dict(threshold_logS=thr,minimum_gain_factor=10**thr,category=category,
        number_of_systems=len(g),count_predicted_peak=int(group_pred.sum()),share_predicted_peak=float(group_pred.mean()),
        interpretation='False positives against narrative no-peak description' if category=='methanol-ethanol' else 'Positive detections against narrative peak description'))
  table=pd.DataFrame(rows)
  table.to_csv(out/'threshold_sensitivity.csv',index=False)
  fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),dpi=170)
  for kind,label,sty in [('alcohol-ketone','Alcohol–ketone (8 systems)','o-'),('methanol-ethanol','Methanol–ethanol (2 systems)','s--')]:
    dat=table[table.category.eq(kind)]
    axes[0].plot(dat.threshold_logS,dat.share_predicted_peak*100,sty,label=label,linewidth=2,markersize=5)
  axes[0].set(xlabel='Required predicted gain above best pure solvent (log10)',ylabel='Curves classified with a peak (%)',title='Classification depends on threshold',ylim=(-4,104));axes[0].grid(alpha=.2);axes[0].legend(loc='best',fontsize=8)
  xx=np.arange(len(bycurve)); colors=['tab:orange' if s=='methanol-ethanol' else 'tab:blue' for s in bycurve.category]
  axes[1].bar(xx,bycurve.grid_direction_changes,color=colors)
  axes[1].set_xticks(xx,[s[:3]+' '+('Et' if a=='ethanol' else 'Me')+'+'+b for s,a,b in zip(bycurve.solute,bycurve.solvent1,bycurve.solvent2)],rotation=65,ha='right',fontsize=7)
  axes[1].set(ylabel='Slope-sign reversals on the 0.1 grid',title='Frozen predicted-curve irregularity',ylim=(0,6));axes[1].grid(axis='y',alpha=.2)
  fig.suptitle('CHEM-02: post-freeze exploratory shape audit (no Table S3 labels)',fontsize=11)
  fig.tight_layout();fig.savefig(out/'fig20_threshold_and_shape.png',dpi=190);plt.close(fig)
  input_sha=hashlib.sha256(src.read_bytes()).hexdigest()
  stats={'frozen_input_sha256':input_sha,'n_rows':len(d),'n_systems':len(bycurve),'all_external_numeric_labels_absent':True,
   'n_alcohol_ketone':int(bycurve.category.eq('alcohol-ketone').sum()),'n_methanol_ethanol':int(bycurve.category.eq('methanol-ethanol').sum()),
   'median_sign_changes':float(bycurve.grid_direction_changes.median()),'range_sign_changes':[int(bycurve.grid_direction_changes.min()),int(bycurve.grid_direction_changes.max())],
   'mean_peak_margin_alcohol_ketone':float(bycurve[bycurve.category.eq('alcohol-ketone')].logS_peak_margin_vs_best_endpoint.mean()),
   'source':'Frozen unlabeled model grid plus qualitative peak/no-peak assignments from Montenegro et al 2026 journal narrative',
   'do_not_claim':'No experimental numeric external MAE/RMSE, no prospective blinding, and no physically impossible conclusion from wiggles'}
  (out/'phase20_numeric_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
  print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
