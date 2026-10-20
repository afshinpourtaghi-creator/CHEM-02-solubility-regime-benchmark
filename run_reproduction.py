#!/usr/bin/env python3
"""Portable entry point for CHEM-02 primary analyses.
EXAMPLE: python run_reproduction.py --raw /path/MixtureSolDB.csv --out /path/run --stage prepare
Heavy model stages can take hours; expected outputs for verification are archived separately.
"""
import argparse,subprocess,sys,json
from pathlib import Path
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--raw',type=Path,help='Required for all stages except verify_archive')
ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--stage',choices=['prepare','raw_audit','phase12','phase13','phase14','classical','verify_archive','all'],default='prepare')
ap.add_argument('--phase13-splits',default='source_disjoint:1001,source_disjoint:1002,source_disjoint:1003,joint_chemical_OOD:1001,joint_chemical_OOD:1002,joint_chemical_OOD:1003')
ap.add_argument('--phase12-seeds',default='901,902')
ap.add_argument('--reference-metadata',type=Path,help='Optional independently retained reference metadata for raw_audit; default compares two code implementations')
ap.add_argument('--threads',type=int,default=3)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if a.stage != 'verify_archive' and (a.raw is None or not a.raw.is_file()): ap.error('A valid --raw CSV file is required for this stage')
metadata=a.out/'harmonized_metadata.csv'
def run(script,*args):
 c=[sys.executable,str(P/script),*map(str,args)]
 print('RUN:',*c,flush=True)
 subprocess.run(c,check=True)
if a.stage in ('prepare','all') or (a.stage not in ('verify_archive',) and not metadata.is_file()):
 run('prepare_metadata.py','--raw',a.raw,'--out',metadata)
if a.stage in ('raw_audit','all'):
 run('phase17_independent_rebuild.py','--raw',a.raw,'--historical',(a.reference_metadata or metadata),'--out',a.out/'audit')
if a.stage in ('phase12','all'):
 run('phase12_nested_benchmark.py','--raw',a.raw,'--input',metadata,'--out',a.out/'phase12','--seeds',a.phase12_seeds,'--threads',a.threads)
if a.stage in ('phase13','all'):
 run('phase13_hard_constraint_nested.py','--raw',a.raw,'--metadata',metadata,'--out',a.out/'phase13','--splits',a.phase13_splits,'--threads',a.threads)
if a.stage in ('phase14','all'):
 run('phase14_endpoint_ablation.py','--raw',a.raw,'--metadata',metadata,'--out',a.out/'phase14')
if a.stage in ('classical','all'):
 run('phase16_classical.py','--raw',a.raw,'--metadata',metadata,'--out',a.out/'classical')
if a.stage in ('verify_archive','all'):
 run('verify_aggregates.py')
print(json.dumps({'stage':a.stage,'status':'COMPLETED','note':'Completing a script does not establish external prospective validation.'},indent=2))
