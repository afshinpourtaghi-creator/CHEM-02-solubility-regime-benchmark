"""Verify archived CHEM-02 aggregate results without redistributing raw data.
Run: python verify_aggregates.py from reproducibility/ directory.
Does not replace running the original model training.
"""
from pathlib import Path
import csv,json,statistics,math,hashlib
r=Path(__file__).parent/'results'
def rows(f):
 with open(r/f,encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
summary=rows('phase13_3seed_summary.csv');table={(x['protocol'],x['method']):float(x['mean_macro']) for x in summary}
assert abs(table[('source_disjoint','direct_all')]-0.5239931213681084)<1e-10
assert abs(table[('source_disjoint','gated_vanthoff')]-0.5820052000009318)<1e-10
assert abs(table[('joint_chemical_OOD','direct_all')]-0.7124977318036644)<1e-10
assert abs(table[('joint_chemical_OOD','gated_vanthoff')]-0.7232078019296244)<1e-10
classic=rows('local_threepoint_summary.csv'); m={x['method']:float(x['system_macro_MAE']) for x in classic};assert math.isclose(m['JA_3term_curve_MAE'],0.04758915480806348,abs_tol=1e-10)
p14=rows('fast_comparison_metrics.csv');d=[float(x['system_macro_MAE']) for x in p14 if x['model']=='direct_250'];g=[float(x['system_macro_MAE']) for x in p14 if 'OOF_endpoint' in x['model']];assert len(d)==len(g)==3;assert abs(statistics.mean(d)-0.708865)<0.001 and abs(statistics.mean(g)-0.761451)<0.001
fr=rows('FROZEN_external_grid_predictions.csv');assert len(fr)==110
print(json.dumps({'status':'PASS','source_disjoint':{'direct':table[('source_disjoint','direct_all')],'gated':table[('source_disjoint','gated_vanthoff')]},'joint_chemical_OOD':{'direct':table[('joint_chemical_OOD','direct_all')],'gated':table[('joint_chemical_OOD','gated_vanthoff')]},'local_JA_3term_system_macro_MAE':m['JA_3term_curve_MAE'],'triple_disjoint':{'direct':statistics.mean(d),'gated':statistics.mean(g)},'frozen_prediction_rows':len(fr)},indent=2))
