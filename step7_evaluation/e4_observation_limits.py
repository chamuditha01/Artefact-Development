import csv, re, datetime as dt, statistics as st
from collections import defaultdict
import os
D=lambda s: dt.date.fromisoformat(s)
iso=re.compile(r'\d{4}-\d{2}-\d{2}')
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
rec={}
for line in open(os.path.join(data_dir, 'ref_labelling_record.md'),encoding='utf-8'):
    if not re.match(r'\|\s*\d+\s*\|\s*REF_',line): continue
    c=[x.strip() for x in line.split('|')]
    eid=c[2]; tf=iso.search(c[3]); tl=iso.search(c[4]);
    cls=c[6]
    cl='uncertain' if 'uncertain' in cls.lower() else re.sub(r'[*]','',cls).split(' ')[0].strip()
    rec[eid]=dict(tf=tf.group() if tf else None, tl=tl.group() if tl else None, cls=cl, raw=c[5], conf=c[9] if len(c)>9 else '')
print(len(rec),"events parsed")
ts=defaultdict(list)
for r in csv.DictReader(open(os.path.join(data_dir, 'REF_all_timeseries.csv'))):
    if float(r['valid_pct'])>=60: ts[r['event_id']].append(D(r['date']))
for k in ts: ts[k].sort()
feat={r['event_id']:r for r in csv.DictReader(open(os.path.join(data_dir, 'features_REF_merged.csv')))}
# interval widths
rows=[]
for e,d in rec.items():
    if d['tf'] and d['tl']:
        w=(D(d['tl'])-D(d['tf'])).days
        rows.append((e,d['cls'],w))
print("with ISO dates both:",len(rows))
ws=[w for _,_,w in rows]
print("interval days: n=%d min=%d median=%s max=%d"%(len(ws),min(ws),st.median(ws),max(ws)))
print("<=60:",sum(1 for w in ws if w<=60),">60:",sum(1 for w in ws if w>60), [ (e,w) for e,_,w in rows if w>60])
by=defaultdict(list)
for e,c,w in rows: by[c].append(w)
for c,l in by.items(): print(c,len(l),"median",st.median(l),"min",min(l),"max",max(l))
# class counts overall
cc=defaultdict(int)
for e,d in rec.items(): cc[d['cls']]+=1
print(dict(cc))
# clear obs in windows after t_loss
out=[]
for e,d in rec.items():
    if not d['tl'] or e not in ts: continue
    tl=D(d['tl'])
    n60=sum(1 for x in ts[e] if tl<=x<=tl+dt.timedelta(days=60))
    n12=sum(1 for x in ts[e] if tl<=x<=tl+dt.timedelta(days=365))
    f=feat.get(e)
    out.append((e,n60,n12,f['n_obs_s1'] if f else None,f['n_obs_s2'] if f else None))
print("event, n60(recomputed), n12(recomputed), n_obs_s1(features), n_obs_s2(features)")
for o in sorted(out): print(o)
n60=[o[1] for o in out]; n12=[o[2] for o in out]
print("Stage1 window clear obs: n=%d min=%d median=%s max=%d; zero in %d events"%(len(n60),min(n60),st.median(n60),max(n60),sum(1 for x in n60 if x==0)))
print("Stage2 window clear obs: n=%d min=%d median=%s max=%d"%(len(n12),min(n12),st.median(n12),max(n12)))
fs1=[float(o[3]) for o in out if o[3] is not None]; fs2=[float(o[4]) for o in out if o[4] is not None]
print("features file n_obs_s1: n=%d min=%d median=%s max=%d; <=1: %d; <=2: %d"%(len(fs1),min(fs1),st.median(fs1),max(fs1),sum(1 for x in fs1 if x<=1),sum(1 for x in fs1 if x<=2)))
print("features file n_obs_s2: n=%d min=%d median=%s max=%d"%(len(fs2),min(fs2),st.median(fs2),max(fs2)))
# worst-case sliding windows (reproduce decisions 2026-09-28)
for L in (30,60,90,120,150,180):
    worst=None
    for e,dates in ts.items():
        a,b=dates[0],dates[-1]; s=a; mn=None
        while s+dt.timedelta(days=L)<=b:
            n=sum(1 for x in dates if s<=x<s+dt.timedelta(days=L)); mn=n if mn is None else min(mn,n); s+=dt.timedelta(days=30)
        if mn is not None: worst=mn if worst is None else min(worst,mn)
    print("window",L,"days: worst-case min clear obs across events =",worst)
# typical (median over windows) per window length
for L in (60,90):
    allc=[]
    for e,dates in ts.items():
        a,b=dates[0],dates[-1]; s=a
        while s+dt.timedelta(days=L)<=b:
            allc.append(sum(1 for x in dates if s<=x<s+dt.timedelta(days=L))); s+=dt.timedelta(days=30)
    allc.sort(); print("window",L,"median count",st.median(allc),"p10",allc[int(.1*len(allc))],"share of windows with 0:",sum(1 for x in allc if x==0)/len(allc))
# monthly clear obs per year (all events)
m=defaultdict(int)
for e,dates in ts.items():
    for x in dates: m[x.month]+=1
print("clear obs by month",dict(sorted(m.items())))
