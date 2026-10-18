import csv, re, json, time, datetime as dt, statistics as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step5_classification'))
from classifiers import *
t0=time.time()
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
rows=[]
for r in csv.DictReader(open(os.path.join(data_dir, 'features_REF_merged.csv'))):
    for k in ["shape_compactness","slope_deg","dist_to_prior_loss_m","trend_nbr_per_day","wobble_std","cycle_count_provisional","n_obs_s1","n_obs_s2","n_obs_baseline"]:
        r[k]=float(r[k])
    rows.append(r)
def t3(c): return "cultivation" if c in("cultivation","cultivation_pasture") else c
res=[r for r in rows if r['known_class'] in("cultivation","cultivation_pasture","permanent_encroachment","natural")]
print("resolved",len(res))
# Stage1
s1=[(r['event_id'],t3(r['known_class']),stage1_classify(r).label) for r in res]
c1=sum(("candidate_natural" if t=="natural" else "candidate_human")==p for _,t,p in s1)
print("Stage1",c1,len(res))
human=[r for r in res if t3(r['known_class'])!="natural"]
X=np.array([[r[f] for f in STAGE2_FEATURES] for r in human]); y=np.array([1 if r['known_class']=="permanent_encroachment" else 0 for r in human]); ids=[r['event_id'] for r in human]
pred={};prob={}
for tr,te in LeaveOneOut().split(X):
    sc=StandardScaler().fit(X[tr]); m=LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]),y[tr])
    p=m.predict_proba(sc.transform(X[te]))[0][1]; prob[ids[te[0]]]=p; pred[ids[te[0]]]=int(p>=0.5)
c2=sum(pred[i]==yy for i,yy in zip(ids,y)); print("Stage2 LOOCV",c2,len(y))
# confusion
tp=sum(1 for i,yy in zip(ids,y) if yy==1 and pred[i]==1); fn=sum(1 for i,yy in zip(ids,y) if yy==1 and pred[i]==0)
fp=sum(1 for i,yy in zip(ids,y) if yy==0 and pred[i]==1); tn=sum(1 for i,yy in zip(ids,y) if yy==0 and pred[i]==0)
print("S2 confusion TP(PE->PE)",tp,"FN",fn,"FP",fp,"TN",tn, "prec",tp/(tp+fp),"recall",tp/(tp+fn))
print("majority baseline",(y==0).sum(),len(y),(y==0).mean())
# by confidence band
from collections import defaultdict
b=defaultdict(lambda:[0,0])
for i,yy in zip(ids,y):
    bb=band(prob[i]); b[bb][1]+=1; b[bb][0]+= (pred[i]==yy)
print("by band",dict(b))
for i,yy in zip(ids,y): print(i,"true",yy,"p=%.3f"%prob[i],band(prob[i]),"OK" if pred[i]==yy else "MISS")
# provenance split
byprov=defaultdict(lambda:[0,0])
for r,yy in zip(human,y):
    k=r['date_reliability']; byprov[k][1]+=1; byprov[k][0]+= (pred[r['event_id']]==yy)
print("by date_reliability",dict(byprov))
# end to end
sc=StandardScaler().fit(X); full=LogisticRegression(max_iter=1000).fit(sc.transform(X),y)
e2e=0
for r in res:
    t=t3(r['known_class']); s=stage1_classify(r).label
    if s=="candidate_natural": fp_="natural"
    elif r['event_id'] in pred: fp_="permanent_encroachment" if pred[r['event_id']] else "cultivation"
    else:
        x=sc.transform([[r[f] for f in STAGE2_FEATURES]]); fp_="permanent_encroachment" if full.predict(x)[0] else "cultivation"
    e2e+= fp_==t
print("E2E",e2e,len(res))
# E2 progressive: events with both rounds
print("Stage1 calls human:",sum(1 for _,_,p in s1 if p=="candidate_human"),"; Stage1 contradicted by hand label:",sum(1 for _,t,p in s1 if (p=="candidate_human")!=(t!="natural")))
print("runtime s",time.time()-t0)
# Stage1 feature distributions by class (for completeness)
for cl in("cultivation","permanent_encroachment","natural"):
    g=[r for r in res if t3(r['known_class'])==cl]
    print(cl,len(g),"median slope",st.median(r['slope_deg'] for r in g),"median compact",st.median(r['shape_compactness'] for r in g))
