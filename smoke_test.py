"""Fast dependency and small real-data feature/model smoke test; NOT a scientific holdout."""
import argparse,hashlib
from pathlib import Path
import pandas as pd,numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from lightgbm import LGBMRegressor
ap=argparse.ArgumentParser();ap.add_argument('--metadata',required=True);a=ap.parse_args()
d=pd.read_csv(a.metadata,nrows=1400)
assert {'solute','A','B','z','y','T','RecordKey','system'}<=set(d.columns)
assert d.RecordKey.is_unique
all_s=sorted(set(d.solute)|set(d.A)|set(d.B));gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=128)
f={}
for s in all_s:
 m=Chem.MolFromSmiles(s);assert m is not None
 v=np.zeros(128,dtype=np.int8);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),v)
 f[s]=v
s=np.stack([f[x] for x in d.solute]); aa=np.stack([f[x] for x in d.A]);bb=np.stack([f[x] for x in d.B]);z=d.z.to_numpy();T=d['T'].to_numpy()
X=np.column_stack([s,aa*z[:,None]+bb*(1-z)[:,None],z,T/300,300/T]).astype('float32')
m=LGBMRegressor(n_estimators=10,num_leaves=7,random_state=42,verbosity=-1,n_jobs=2);m.fit(X[:1100],d.y.to_numpy()[:1100]);pred=m.predict(X[1100:]);assert np.isfinite(pred).all()
print('SMOKE PASS: raw-derived schema + RDKit features + LightGBM training/prediction (NOT validation)')
