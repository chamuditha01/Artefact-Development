"""
auto_dating.py - automatic loss-date interval for one event, checked against hand dates.

PARAMETERS WERE FIXED BEFORE THE FIRST RUN (2026-10-07). Do not change them to
make the agreement look better. If a parameter changes, log the change and the
reason in decisions.md and report BOTH runs.

RULE (a coding of the reading guide in decisions.md 2026-09-22, not new science):
  1. Use clear observations only: valid_pct >= MIN_VALID_PCT.
  2. Search window = 1 Jan (H-1) to 31 Dec (H), where H is the Hansen loss year.
     H-1 is included because Hansen's year was off by about a year for REF_049.
  3. For each observation i in the window, in date order:
       baseline B = median NBR of clear observations in the BASELINE_DAYS before i
       (needs at least MIN_BASE_OBS observations, and B > 0, else skip i).
       i is a drop if NBR_i < FRACTION * B AND the next (K-1) clear observations
       are also below FRACTION * B.
  4. The first drop is t_loss. t_forest = the last earlier observation with
     NBR >= FRACTION * B. If there is none, the event fails ("no_t_forest").
  5. Interval = [t_forest, t_loss]. Width > T7_DAYS -> "wide" (would be uncertain).
  No drop found -> "no_drop". Nothing is interpolated. Nothing is tuned.

AGREEMENT TEST, fixed in advance:
  PRIMARY: |auto t_loss - hand t_loss| <= AGREE_DAYS.
  SECONDARY: auto interval overlaps hand interval.
Hand dates: feature_extraction_batch_REF.js (29 events). REF_040 has none and is
reported separately; the expected result is "no usable interval".

Uses only the standard library.
"""
import csv, sys, statistics, datetime as dt
from collections import defaultdict

MIN_VALID_PCT = 60
FRACTION      = 0.75
K             = 2
BASELINE_DAYS = 365
MIN_BASE_OBS  = 3
T7_DAYS       = 60
AGREE_DAYS    = 30
WINDOW_BACK_YEARS = 1   # run 1 = 1 (search H-1..H). v2 = 0 (search H only). Set by --back

HAND = {  # event: (t_forest, t_loss)
'REF_004':('2022-02-04','2022-03-11'),'REF_039':('2023-02-14','2023-03-06'),
'REF_055':('2023-06-16','2023-07-16'),'REF_026':('2024-06-30','2024-07-20'),
'REF_058':('2023-01-20','2023-03-26'),'REF_053':('2024-05-26','2024-05-31'),
'REF_005':('2022-02-01','2022-03-13'),'REF_028':('2023-06-16','2023-08-05'),
'REF_041':('2022-07-01','2022-07-26'),'REF_013':('2023-03-06','2023-03-16'),
'REF_020':('2021-06-29','2021-08-23'),'REF_008':('2020-01-31','2020-02-10'),
'REF_021':('2023-03-01','2023-03-11'),'REF_016':('2022-04-25','2022-04-30'),
'REF_044':('2019-11-17','2019-12-27'),'REF_047':('2022-03-11','2022-03-16'),
'REF_034':('2023-06-04','2023-06-24'),'REF_049':('2020-02-20','2020-03-01'),
'REF_052':('2022-07-01','2022-07-26'),'REF_003':('2024-01-22','2024-01-27'),
'REF_018':('2023-07-21','2023-08-25'),'REF_014':('2024-08-14','2024-09-18'),
'REF_023':('2021-01-30','2021-02-14'),'REF_012':('2021-06-09','2021-08-23'),
'REF_007':('2020-02-02','2020-03-18'),'REF_042':('2024-05-14','2024-06-23'),
'REF_033':('2022-04-25','2022-04-30'),'REF_031':('2022-06-26','2022-08-10'),
'REF_025':('2016-11-01','2017-06-01'),
}
D = lambda s: dt.date.fromisoformat(s)

def auto_interval(obs, hansen_year):
    """obs: sorted list of (date, nbr), clear only. Returns dict."""
    ws, we = dt.date(hansen_year-WINDOW_BACK_YEARS,1,1), dt.date(hansen_year,12,31)
    for i,(d,v) in enumerate(obs):
        if d < ws or d > we: continue
        base = [x for (dd,x) in obs if d-dt.timedelta(days=BASELINE_DAYS) <= dd < d]
        if len(base) < MIN_BASE_OBS: continue
        B = statistics.median(base)
        if B <= 0: continue
        thr = FRACTION*B
        if v >= thr: continue
        nxt = obs[i+1:i+K]
        if len(nxt) < K-1 or any(x >= thr for (_,x) in nxt): continue
        prior = [dd for (dd,x) in obs[:i] if x >= thr]
        if not prior: return {'status':'no_t_forest','t_loss':d,'baseline':B}
        tf = prior[-1]
        w = (d-tf).days
        return {'status':'ok' if w<=T7_DAYS else 'wide','t_forest':tf,'t_loss':d,'width':w,'baseline':B}
    return {'status':'no_drop'}

def load_hand_csv(path):
    """hand_status in {dated, no_drop, undateable}; dates only for 'dated'."""
    out={}
    for r in csv.DictReader(open(path)):
        out[r['event_id']]=r
    return out

def main(path_ts, out_csv):
    ev = defaultdict(list); hy = {}
    for r in csv.DictReader(open(path_ts)):
        if float(r['valid_pct']) < MIN_VALID_PCT: continue
        ev[r['event_id']].append((D(r['date']), float(r['NBR'])))
        hy[r['event_id']] = int(float(r['hansen_year']))
    rows=[]
    for e in sorted(ev):
        obs = sorted(dict(ev[e]).items())          # one value per date
        a = auto_interval(obs, hy[e])
        row = {'event_id':e,'hansen_year':hy[e],'auto_status':a['status'],
               'auto_t_forest':a.get('t_forest',''),'auto_t_loss':a.get('t_loss',''),
               'auto_width':a.get('width','')}
        h = HAND.get(e)
        if h:
            row.update(hand_t_forest=h[0], hand_t_loss=h[1],
                       hand_width=(D(h[1])-D(h[0])).days)
            if a.get('t_loss'):
                row['d_t_loss_days']=(a['t_loss']-D(h[1])).days
                if a.get('t_forest'):
                    row['overlap']= int(a['t_forest']<=D(h[1]) and D(h[0])<=a['t_loss'])
        rows.append(row)
    cols=['event_id','hansen_year','hand_t_forest','hand_t_loss','hand_width','auto_status',
          'auto_t_forest','auto_t_loss','auto_width','d_t_loss_days','overlap']
    w=csv.DictWriter(open(out_csv,'w',newline=''),fieldnames=cols,extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
    hand=[r for r in rows if r['event_id'] in HAND]
    n=len(hand)
    ok=[r for r in hand if r['auto_status']=='ok']
    wide=[r for r in hand if r['auto_status']=='wide']
    nodrop=[r for r in hand if r['auto_status'] in ('no_drop','no_t_forest')]
    got=[r for r in hand if 'd_t_loss_days' in r]
    agree=[r for r in got if abs(r['d_t_loss_days'])<=AGREE_DAYS]
    ov=[r for r in got if r.get('overlap')==1]
    print(f'window: H-{WINDOW_BACK_YEARS}..H   hand-dated events: {n}')
    print(f'auto status  ok(<= {T7_DAYS}d): {len(ok)}  wide: {len(wide)}  no drop/no t_forest: {len(nodrop)}')
    print(f'auto t_loss found: {len(got)}; within {AGREE_DAYS} d of hand t_loss: {len(agree)} of {n}')
    print(f'interval overlaps hand interval: {len(ov)} of {len(got)} with an interval')
    if got:
        ds=sorted(abs(r['d_t_loss_days']) for r in got)
        print('abs diff in t_loss (days), sorted:', ds)
    r40=[r for r in rows if r['event_id']=='REF_040']
    print('REF_040 (no hand date):', r40[0]['auto_status'] if r40 else 'not in file')
    print('wrote', out_csv)

if __name__=='__main__':
    args=sys.argv[1:]
    if '--back' in args:
        i=args.index('--back'); WINDOW_BACK_YEARS=int(args[i+1]); del args[i:i+2]
    if len(args)>=3:
        # fresh-set mode: python auto_dating.py ts.csv out.csv hand_fresh.csv
        HANDCSV=load_hand_csv(args[2])
        HAND.clear()
        for e,r in HANDCSV.items():
            if r['hand_status']=='dated': HAND[e]=(r['t_forest'],r['t_loss'])
        FRESH_STATUS={e:r['hand_status'] for e,r in HANDCSV.items()}
        main(args[0], args[1])
        # false-confident rate on events with no real drop / undateable
        import csv as _c
        rows=list(_c.DictReader(open(args[1])))
        bad=[r for r in rows if FRESH_STATUS.get(r['event_id'],'') in ('no_drop','undateable')]
        conf=[r for r in bad if r['auto_status']=='ok']
        print(f"hand says no drop / undateable: {len(bad)}; rule still gave a short interval: {len(conf)}")
    else:
        main(args[0], args[1])
