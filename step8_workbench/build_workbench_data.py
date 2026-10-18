"""
build_workbench_data.py  -  2026-10-08
Builds workbench_data.js for workbench.html from files already in the project.
Inputs (all local):
  ../../REF_all_timeseries.csv or ../REF series      NBR series, 30 REF events
  ../fresh/REF_fresh_timeseries_g*.csv               NBR series, 12 fresh events
  features_REF_merged.csv                            features, 29 evaluated events
  features_FRESH_group*.csv (optional)               features, 8 fresh events (GEE output)
  ../hand_dates_fresh.csv                            hand dates, fresh events
  event_geometry.geojson (optional)                  footprints
Rules this file enforces:
  * Stage 2 on the 19 labelled Stage-2 events is LEAVE-ONE-OUT, never in-sample.
  * Fresh events: model output, not evaluated. Final-fit Stage 2 (they are not in training).
  * Events with no usable date get NO call.
  * The reference label (answer key) is carried only for the 30 REF events and is hidden by default in the page.
"""
import csv, glob, json, os, sys, statistics, datetime as dt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import classifiers as C
import auto_dating as AD
AD.WINDOW_BACK_YEARS = 0          # rule v2, the better of the two on the fresh test

def rd(p): return list(csv.DictReader(open(p)))
def fnum(x):
    try: return float(x)
    except: return None

# ---- series
series, site, hy = {}, {}, {}
ref_series_path = [p for p in (os.path.join(HERE,'..','..','REF_all_timeseries.csv'),
                               os.path.join(HERE,'..','REF_all_timeseries.csv')) if os.path.exists(p)][0]
for path in [ref_series_path] + sorted(glob.glob(os.path.join(HERE,'..','fresh','REF_fresh_timeseries_g*.csv'))):
    for r in rd(path):
        if float(r['valid_pct']) < AD.MIN_VALID_PCT: continue
        e = r['event_id']
        series.setdefault(e, {})[r['date']] = float(r['NBR'])
        site[e] = r['site']; hy[e] = int(float(r['hansen_year']))
# fresh Hansen years come from the g-files, which are correct (checked 2026-10-08)
# Maduroad: all 104 patches. Existing series for already-held events win; only NEW patches are added.
MAD_ADDED = []
for path in sorted(glob.glob(os.path.join(HERE,'..','mad','MAD_timeseries_c*.csv'))):
    for r in rd(path):
        e = r['event_id']
        if e in series and e not in MAD_ADDED: continue       # keep the already-held series
        if float(r['valid_pct']) < AD.MIN_VALID_PCT: continue
        series.setdefault(e, {})[r['date']] = float(r['NBR'])
        site[e] = r['site']; hy[e] = int(float(r['hansen_year']))
        if e not in MAD_ADDED: MAD_ADDED.append(e)

# ---- hand dates
hand = {e: dict(t_forest=v[0], t_loss=v[1], status='dated', source='read by eye from the NBR chart (researcher)')
        for e, v in AD.HAND.items()}
for r in rd(os.path.join(HERE,'..','hand_dates_fresh.csv')):
    if r['hand_status'] == 'dated':
        hand[r['event_id']] = dict(t_forest=r['t_forest'], t_loss=r['t_loss'], status='dated',
            source='first-pass read by Claude, confirmed by researcher 2026-10-08', note=r['notes'])
    else:
        hand[r['event_id']] = dict(status='undateable', source='first-pass read, confirmed', note=r['notes'])
hand['REF_040'] = dict(status='undateable', source='researcher: three-way date mismatch',
                       note='No usable t_forest / t_loss (labelled uncertain, date-unreliable).')

# ---- reference labels (answer key) and features
ref_rows = rd(os.path.join(HERE,'features_REF_merged.csv'))
NUM = ['area_ha','i_pre','n_obs_baseline','shape_compactness','slope_deg','dist_to_prior_loss_m',
       'n_obs_s1','early_spectral_change','n_obs_s2','trend_nbr_per_day','wobble_std','cycle_count_provisional']
feats, known = {}, {}
for r in ref_rows:
    feats[r['event_id']] = {k: fnum(r[k]) for k in NUM}
    known[r['event_id']] = r['known_class']
known['REF_040'] = 'uncertain (date-unreliable)'
fresh_feat_files = sorted(glob.glob(os.path.join(HERE,'features_FRESH_group*.csv')))
for p in fresh_feat_files:
    for r in rd(p):
        feats[r['event_id']] = {k: fnum(r[k]) for k in NUM}
        feats[r['event_id']]['date_reliability'] = r.get('date_reliability')

# Maduroad rule-dated features (dates from auto_dating rule v2; NOT hand-checked)
for p in sorted(glob.glob(os.path.join(HERE,'..','mad','features_MAD_group*.csv'))):
    for r in rd(p):
        feats[r['event_id']] = {k: fnum(r[k]) for k in NUM}
        feats[r['event_id']]['date_reliability'] = r.get('date_reliability')
RULE_DATED_FEATS = {e for e in feats if (feats[e].get('date_reliability') or '') == 'rule_dated_unverified'}

# ---- Stage 2 models
S2F = C.STAGE2_FEATURES
labelled = [dict(r, **{k: fnum(r[k]) for k in NUM}) for r in ref_rows]
final_fit = C.Stage2Classifier.fit_from_labelled_events(labelled)
elig = [r for r in labelled if r['known_class'] in ('cultivation','cultivation_pasture','permanent_encroachment')]
loo = {}
for r in elig:
    m = C.Stage2Classifier.fit_from_labelled_events([x for x in labelled if x['event_id'] != r['event_id']])
    loo[r['event_id']] = m.classify({k: fnum(r[k]) for k in S2F})

def call_for(e):
    f = feats.get(e)
    if f is None: return None
    if any(f.get(k) is None for k in ('shape_compactness','slope_deg','dist_to_prior_loss_m')):
        return {'note': 'Stage 1 feature missing (for example no prior loss within 10 km)'}
    s1 = C.stage1_classify(f)
    out = {'stage1': dict(label=s1.label, confidence=s1.confidence, legs=s1.legs_passed, note=s1.evidence_note),
           'stage2': None}
    if s1.label == 'candidate_human' and all(f.get(k) is not None for k in S2F):
        if e in loo: s2, basis = loo[e], 'leave-one-out (this event left out of the fit)'
        else:        s2, basis = final_fit.classify(f), 'model fitted on the 19 labelled events; this event is not one of them'
        out['stage2'] = dict(label=s2.label, p_permanent=round(s2.probability_permanent,3),
                             confidence=s2.confidence, basis=basis)
    return out

# ---- Stage 1 on undated Maduroad patches (static features only; no dates used)
S1ONLY = {}
_sp = os.path.join(HERE, '..', 'mad', 'MAD_stage1_static.csv')
if os.path.exists(_sp):
    for r in csv.DictReader(open(_sp)):
        try: f = {k: float(r[k]) for k in ('shape_compactness','slope_deg','dist_to_prior_loss_m')}
        except (ValueError, TypeError): continue
        s1 = C.stage1_classify(f)
        S1ONLY[r['event_id']] = dict(label=s1.label, legs=s1.legs_passed, note=s1.evidence_note)

# ---- geometry (optional)
geom = {}
for gname in ('event_geometry.geojson', 'event_geometry_maduroad.geojson'):
    gp = os.path.join(HERE, gname)
    if os.path.exists(gp):
        for ft in json.load(open(gp))['features']:
            geom.setdefault(ft['properties']['queue_id'], ft['geometry'])

EVAL = set(AD.HAND) 
FRESH = [f'REF_{n:03d}' for n in range(59, 71)]
order = sorted(AD.HAND) + ['REF_040'] + FRESH + sorted(x for x in MAD_ADDED if x not in AD.HAND and x not in FRESH and x != 'REF_040')
events = []
for e in order:
    obs = sorted(series[e].items())
    if not obs: continue
    h = hand.get(e)
    rule = AD.auto_interval([(AD.D(d), v) for d, v in obs], hy[e])
    rule = {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in rule.items()}
    rule_dated = (e in RULE_DATED_FEATS and rule.get('status') == 'ok' and e not in hand)
    if rule_dated:
        h = dict(t_forest=rule['t_forest'], t_loss=rule['t_loss'], status='dated', rule_dated=True,
                 source='set by the automatic rule (v2), NOT checked by a person')
    status = 'evaluated' if e in EVAL else ('fresh' if e in FRESH else 'no_dates')
    if e == 'REF_040': status = 'evaluated_no_dates'
    if e in MAD_ADDED and e not in EVAL and e not in FRESH and e != 'REF_040': status = 'maduroad_new'
    num = int(e.split('_')[1])
    screening = ('confirmed' if (e in AD.HAND or e == 'REF_040') else
                 ('screened_not_confirmed' if num <= 58 else 'not_screened'))
    ev = dict(id=e, site=site[e], hansen_year=hy[e], status=status, screening=screening, hand=h, rule=rule,
              series=[[d, round(v, 4)] for d, v in obs], features=feats.get(e),
              call=call_for(e) if (h and h['status'] == 'dated') else None,
              ref_label=known.get(e) if e in known else None, geometry=geom.get(e),
              s1_only=(S1ONLY.get(e) if status == 'maduroad_new' else None), rule_dated=rule_dated)
    if h and h['status'] == 'dated':
        tl = AD.D(h['t_loss']); tf = AD.D(h['t_forest'])
        ev['windows'] = dict(baseline=[(tf - dt.timedelta(days=365)).isoformat(), h['t_forest']],
                             s1=[h['t_loss'], (tl + dt.timedelta(days=60)).isoformat()],
                             s2=[h['t_loss'], (tl + dt.timedelta(days=365)).isoformat()])
    events.append(ev)

meta = dict(built='2026-10-08', n_events=len(events),
            n_with_call=sum(1 for x in events if x['call']),
            fresh_features_loaded=len(fresh_feat_files) > 0, geometry_loaded=bool(geom),
            stage1_thresholds=C.STAGE1_THRESHOLDS, n_stage2_train=final_fit._n_train)
open(os.path.join(HERE,'workbench_data.js'), 'w').write(
    '// generated by build_workbench_data.py - do not edit\nwindow.WB_DATA = ' + json.dumps(dict(meta=meta, events=events)) + ';\n')
print(meta)
from collections import Counter
print(Counter(x['status'] for x in events))
for x in events:
    c = x['call']
    s1 = c['stage1']['label'] if c and 'stage1' in c else '-'
    s2 = (c['stage2']['label'] + ' %.2f' % c['stage2']['p_permanent']) if c and c.get('stage2') else '-'
    print(x['id'], x['status'], (x['hand'] or {}).get('status'), x['rule']['status'], s1, s2, '| ref:', x['ref_label'])
