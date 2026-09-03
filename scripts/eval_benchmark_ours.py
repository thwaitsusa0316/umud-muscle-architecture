#!/usr/bin/env python3
"""Benchmark UMUD with OUR retrained fascicle model in place of DL_Track's.

Coverage proved a gameable gate -- a model that paints the whole belly solid scores
~1.0 on it. This is the number that cannot be gamed: the actual competition metric
against the expert consensus.
"""
import sys, os, pathlib, warnings, argparse
os.environ.setdefault("TF_USE_LEGACY_KERAS","1"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("OPENCV_LOG_LEVEL","SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src")); sys.path.insert(0,"scripts")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from eval_new_fascicle import load_model, predict_ours
from umud import validation as V, segment as S, geometry as Gm

ap=argparse.ArgumentParser(); ap.add_argument("--thr",type=float,default=0.5); ap.add_argument("--ckpt",default=None); a=ap.parse_args()
model,dev=load_model("unet", a.ckpt); t=V.load()
grays=[]
for p in t.path:
    g=cv2.imread(p, cv2.IMREAD_UNCHANGED); grays.append(g if g.ndim==2 else cv2.cvtColor(g,cv2.COLOR_BGR2GRAY))
apo=[am for am,_ in S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)]
rows=[]
for g,am,(_,r) in zip(grays,apo,t.iterrows()):
    fm=predict_ours(model,dev,g,thr=a.thr)
    res=Gm.analyse(am,fm,float(r.px_per_cm))
    rows.append(dict(image_id=r.image_id,pa_deg=res.pa_deg,fl_mm=res.fl_mm,mt_mm=res.mt_mm))
p=pd.DataFrame(rows).dropna()
print(f"thr={a.thr}  usable {len(p)}/{len(t)}")
if len(p):
    print(f"UMUD = {V.score(p,t):.4f}")
    print(V.report(p,t).round(3).to_string(index=False))
print("\nours-with-DLTrack-fascicles 0.3349 | DL_Track 0.3306 | expert mean 0.3032 | constant 0.7032")
