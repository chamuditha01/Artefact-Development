// ============================================================================
// feature_extraction.js  -  v3  -  2026-09-28
// Automated Stage 1 / Stage 2 feature extraction for the driver classifier (D6:
// hand-set thresholds, not a trained model - decisions.md 2026-09-28).
// One event at a time, same pattern as event_timeseries_v6.js.
//
// NOTE ON THE D6 CITATION ABOVE: this comment (unchanged from the original
// version of this file) cites "decisions.md 2026-09-28" for D6 being decided
// as hand-set thresholds. No such entry exists in decisions.md as of this
// edit (2026-09-28, later) - D6 is still listed open in issues.md's deferred-
// decisions table. Left as-is rather than silently rewritten (this file's
// own rule: mark, don't erase), but flagged here so it isn't taken as
// settled. Either find where this was actually decided and log it, or treat
// D6 as still open.
//
// WINDOWS (decisions.md 2026-09-28):
//   Baseline  : t_forest - [T1]=12 months  ->  t_forest
//   Stage 1   : t_loss  ->  t_loss + [T2a]=60 days
//   Stage 2   : t_loss  ->  t_loss + [T2b]=12 months
// No blackout gap / bounded label window is enforced here - see decisions.md
// 2026-09-28 (later) for why. The label these features get compared against
// (calibration_batch_record.csv / ref_labelling_record.md) was read from the
// FULL record, so it is not independent of these windows for trajectory-only
// events. Report results split by label provenance (structural vs
// trajectory-only), per labelling_protocol.md §3 - this script does not do
// that split itself, it only computes features.
//
// FIREWALL: enforced TWICE per stage - structurally (filterDate bounds the
// ImageCollection before anything touches it) and by an explicit assertion
// that fails loudly (throws, does not just print) if any observation used
// falls outside its window. Per labelling_protocol.md §3: "Enforce with an
// assertion per stage in the extraction code that fails loudly, not with a
// comment."
//
// KNOWN BUG THIS SCRIPT MUST NOT REPEAT (issues.md B14/B15/B23): n_valid and
// n_total must come from SEPARATE reduceRegion/count() calls on the same
// masked-NBR grid, never from one combined .combine() reducer. The stats()
// function below is copied from event_timeseries_v6.js (confirmed clean,
// decisions.md 2026-09-25), not rewritten.
//
// OPEN / NOT DONE IN THIS VERSION:
//   - distance_to_road / distance_to_prior_loss (UPDATED 2026-09-28, later;
//     FIXED 2026-09-28, later still; BUFFER WIDENED then REMOVED 2026-09-28,
//     later still again - see the dated note right above the STEP 4
//     computation block below): no verified GEE roads asset for Sri Lanka
//     has been found, so this no longer waits on one. The original Stage 1
//     feature-list decision (decisions.md 2026-09-28) names the feature as
//     "distance to nearest ROAD OR EXISTING CLEARED/LOSS PATCH" - two
//     alternative signals for the same thing, not one. This script computes
//     the loss-patch half: distance to the nearest prior Hansen loss pixel
//     (v1.13, the asset already verified and used throughout this project -
//     no new sources.md entry needed). See STEP 4 below for the current
//     definition (year cutoff only, no spatial buffer) and decisions.md
//     2026-09-28 (later still again) for why the buffer was tried, then
//     dropped. distance-to-road proper is dropped, not deferred - revisit
//     only if a verified roads asset turns up and it's worth adding as a
//     second, separate feature rather than a replacement.
//   - cycle_count (Stage 2) is a first-pass proxy: count of median-crossings
//     of the clear-observation NBR series within the Stage 2 window. Not
//     validated against the CAL/REF labels yet. Flag as provisional - see
//     decisions.md 2026-09-28 (later), "what I gave up".
//   - trend and wobble are computed client-side (values pulled via getInfo())
//     from a small per-event array, same scale as event_timeseries_v6.js's
//     existing client-side reads - fine at one-event-at-a-time scale, not
//     meant to be batched as-is across 30+ events without re-checking cost.
// ============================================================================

// ### SET PER EVENT - update ALL of these each time ###
var SAMPLE_ASSET = 'projects/gsignin-501704/assets/REF_screening_queue'; // or the CAL asset
var EVENT_ID      = 'REF_001';        // queue_id / cal_id
var T_FOREST       = '2022-02-04';    // from calibration_batch_record.csv / ref_labelling_record.md - already labelled, NOT re-derived here
var T_LOSS         = '2022-03-11';    // ditto
var MIN_VALID_PCT  = 60;              // matches [T6], frozen 2026-09-22

// Window lengths - decisions.md 2026-09-28 (later). Do not change here; change there first.
var T1_MONTHS  = 12;   // baseline
var T2A_DAYS   = 60;   // Stage 1 feature window
var T2B_MONTHS = 12;   // Stage 2 feature window
var T14A_MIN   = 1;    // min clear obs, Stage 1
var T14B_MIN   = 3;    // min clear obs, Stage 2

// dist_to_prior_loss parameters (2026-09-28, later) - see file header and
// STEP 4 below. No SELF_BUFFER_M - removed in this version (see STEP 4
// comment and decisions.md 2026-09-28, later still again).
var HANSEN_ID        = 'UMD/hansen/global_forest_change_2025_v1_13';  // same asset as build_screening_queue.js
var SEARCH_RADIUS_M   = 10000;   // chosen ceiling, not measured; events with nothing prior within this come back null
var SEARCH_RADIUS_PX  = Math.ceil(SEARCH_RADIUS_M / 30);

// ---------------------------------------------------------------------------
// STEP 1 - geometry (same asset-load pattern as event_timeseries_v6.js)
// ---------------------------------------------------------------------------
var coll  = ee.FeatureCollection(SAMPLE_ASSET);
var feat  = ee.Feature(coll.filter(ee.Filter.eq('queue_id', EVENT_ID)).first());
var event = feat.geometry();
print('STEP 1 - event', EVENT_ID, '| area (ha)', event.area({maxError: 1}).divide(10000));

// ---------------------------------------------------------------------------
// STEP 2 - shared cloud-mask / NBR / per-image stats functions
// (COPIED, not rewritten, from event_timeseries_v6.js - see file header)
// ---------------------------------------------------------------------------
function maskClouds(img) {
  var scl = img.select('SCL');
  var keep = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(keep);
}
function addNBR(img) {
  return img.addBands(img.normalizedDifference(['B8', 'B12']).rename('NBR'));
}
function stats(img) {
  var nbr = img.select('NBR');
  var nValid = nbr.reduceRegion({reducer: ee.Reducer.count(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  var nTotal = nbr.unmask(-99).reduceRegion({reducer: ee.Reducer.count(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  var meanNBR = nbr.reduceRegion({reducer: ee.Reducer.mean(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  nValid = ee.Number(ee.Algorithms.If(nValid, nValid, 0));
  nTotal = ee.Number(ee.Algorithms.If(nTotal, nTotal, 1));
  return ee.Feature(null, {
    date: img.date().format('YYYY-MM-dd'), millis: img.date().millis(), NBR: meanNBR,
    n_valid: nValid, n_total: nTotal, valid_pct: nValid.divide(nTotal).multiply(100)
  });
}
function clearObsInRange(startDate, endDate) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(event).filterDate(startDate, endDate)   // <-- STRUCTURAL firewall
    .map(maskClouds).map(addNBR);
  var perImage = ee.FeatureCollection(col.map(stats)).filter(ee.Filter.notNull(['NBR']));
  return perImage.filter(ee.Filter.gte('valid_pct', MIN_VALID_PCT)).sort('millis');
}

// Fails loudly (throws) if anything in `obsFC` falls outside [startMillis, endMillis].
// Belt-and-braces on top of the filterDate() call above, per labelling_protocol.md §3.
function assertWithinWindow(stageLabel, obsFC, startMillis, endMillis) {
  var bad = obsFC.filter(ee.Filter.or(
    ee.Filter.lt('millis', startMillis), ee.Filter.gt('millis', endMillis)
  )).size().getInfo();
  if (bad > 0) {
    throw new Error('FIREWALL VIOLATION - ' + stageLabel + ' - ' + EVENT_ID +
      ' - ' + bad + ' observation(s) fall outside [' + startMillis + ', ' + endMillis + ']. STOP.');
  }
  print(stageLabel + ' - firewall check passed (' + obsFC.size().getInfo() + ' obs, all within window)');
}

// ---------------------------------------------------------------------------
// STEP 3 - baseline (I_pre), for the Stage 1 early-spectral-change feature
// ---------------------------------------------------------------------------
var tForest = ee.Date(T_FOREST);
var tLoss   = ee.Date(T_LOSS);
var baselineStart = tForest.advance(-T1_MONTHS, 'month');
var baselineObs = clearObsInRange(baselineStart, tForest);
assertWithinWindow('BASELINE', baselineObs, baselineStart.millis().getInfo(), tForest.millis().getInfo());
var iPre = baselineObs.aggregate_mean('NBR');
print('STEP 3 - I_pre (baseline mean NBR, ' + T1_MONTHS + 'mo)', iPre);

// ---------------------------------------------------------------------------
// STEP 4 - STAGE 1 features: window = t_loss -> t_loss + [T2a] days
// ---------------------------------------------------------------------------
var s1Start = tLoss;
var s1End   = tLoss.advance(T2A_DAYS, 'day');
var s1Obs   = clearObsInRange(s1Start, s1End);
assertWithinWindow('STAGE 1', s1Obs, s1Start.millis().getInfo(), s1End.millis().getInfo());

var s1Count = s1Obs.size().getInfo();
if (s1Count < T14A_MIN) {
  print('STAGE 1 - WARNING: only ' + s1Count + ' clear obs, below [T14a]=' + T14A_MIN +
    ' - mark uncertain per labelling_protocol.md §6, do not force a value.');
}

// shape + slope are static (geometry / terrain), not time-windowed - computed
// once, not gated by the firewall, but reported alongside the windowed features.
var perimeter = event.perimeter({maxError: 1});
var area = event.area({maxError: 1});
var compactness = ee.Number(4).multiply(Math.PI).multiply(area).divide(perimeter.pow(2)); // Polsby-Popper, 1 = circle
var slopeImg = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003'));
var meanSlopeDeg = slopeImg.reduceRegion({reducer: ee.Reducer.mean(), geometry: event, scale: 30, maxPixels: 1e9}).get('slope');

// dist_to_prior_loss (replaces dist_to_road, 2026-09-28 later - see file
// header). Definition: straight-line distance (m) from the event polygon to
// the nearest OTHER Hansen loss pixel with lossyear STRICTLY BEFORE this
// event's own year. The year cutoff does two jobs at once: (1) excludes this
// event's own footprint, since Hansen tags it with the same year, and
// (2) excludes any loss that happened AFTER this event, which would be
// feature-extraction leakage - at t_loss, this event's own future doesn't
// exist yet. Hansen's lossyear is annual, so this cutoff can't distinguish
// month-level ordering within a single year - a real, stated granularity
// limit, same class of caveat as B19's Hansen-year-mismatch finding.
//
// BUFFER REMOVED (2026-09-28, later still again): earlier versions of this
// script also masked out a buffer around the event's own polygon
// (SELF_BUFFER_M), as a belt-and-braces guard against a Hansen pixel at the
// event's edge disagreeing with the vectorised boundary. That buffer was
// widened once (30m -> 90m) after the batch script showed values clustering
// tightly near the buffer edge for about half the 29 REF events. Widening
// did not remove the clustering, it moved it - by almost exactly the buffer
// increase. That proved the "nearest permitted pixel" for those events is
// not self-noise, it's a real, separate-year Hansen loss patch immediately
// next to the event (most likely incremental expansion of an existing
// clearing). The buffer was suppressing genuine adjacency, not filtering
// noise, so it's removed - the year cutoff alone already does the actual
// job (excluding this event's own footprint). A small resulting distance is
// now treated as real signal, not forced away. See decisions.md 2026-09-28
// (later still again) for the full reasoning.
//
// FIX (2026-09-28, later still, kept from the buffer-era version):
// ee.Image.fastDistanceTransform() does not preserve the input band's name,
// so reading the reduceRegion result by .get('lossmask') threw "Dictionary
// does not contain key". Read by position instead (.values().get(0)), after
// checking the dictionary's size - this also correctly resolves to null
// (not an error) when an event genuinely has no prior loss within the
// search radius. Same pattern build_screening_queue.js's water-distance
// code already uses, for the same reason.
var eventYearCode = tLoss.get('year').subtract(2000);   // Hansen codes: 19=2019 ... 24=2024
var priorLossMask = ee.Image(HANSEN_ID).select('lossyear')
  .gt(0).and(ee.Image(HANSEN_ID).select('lossyear').lt(eventYearCode))
  .rename('lossmask')
  .selfMask();
var lossDistImg = priorLossMask.fastDistanceTransform(SEARCH_RADIUS_PX).sqrt()
  .multiply(ee.Image.pixelArea().sqrt());
var lossDistDict = lossDistImg.reduceRegion({
  reducer: ee.Reducer.min(), geometry: event, scale: 30, maxPixels: 1e9
});
var distToPriorLossM = ee.Algorithms.If(lossDistDict.size().gt(0), lossDistDict.values().get(0), null);

var minNBRInWindow = s1Obs.aggregate_min('NBR');
var earlySpectralChange = ee.Algorithms.If(
  ee.Number(s1Count).gt(0),
  ee.Number(iPre).subtract(ee.Number(minNBRInWindow)),  // positive = drop from baseline
  null
);

print('STAGE 1 FEATURES', ee.Dictionary({
  event_id: EVENT_ID, window_days: T2A_DAYS, n_obs: s1Count,
  shape_compactness: compactness, slope_deg: meanSlopeDeg, dist_to_prior_loss_m: distToPriorLossM,
  early_spectral_change: earlySpectralChange
}));
// distToPriorLossM prints as null when nothing prior falls inside the
// SEARCH_RADIUS_M search radius - that's a real "isolated event" reading,
// not a computation failure, so it's left null rather than forced to a
// sentinel value. Check for it manually per event; not asserted here.

// ---------------------------------------------------------------------------
// STEP 5 - STAGE 2 features: window = t_loss -> t_loss + [T2b] months
// ---------------------------------------------------------------------------
var s2Start = tLoss;
var s2End   = tLoss.advance(T2B_MONTHS, 'month');
var s2Obs   = clearObsInRange(s2Start, s2End);
assertWithinWindow('STAGE 2', s2Obs, s2Start.millis().getInfo(), s2End.millis().getInfo());

var s2Count = s2Obs.size().getInfo();
if (s2Count < T14B_MIN) {
  print('STAGE 2 - WARNING: only ' + s2Count + ' clear obs, below [T14b]=' + T14B_MIN +
    ' - per §6, exclude this event from Stage 2 only, keep it in Stage 1.');
}

// Pull the small per-event array client-side (same scale as event_timeseries_v6.js's
// existing client-side reads) to compute trend / wobble / cycle_count in plain JS.
var s2List = s2Obs.select(['millis', 'NBR']).getInfo().features.map(function (f) {
  return {millis: f.properties.millis, nbr: f.properties.NBR};
});

var trend = null, wobble = null, cycleCount = null;
if (s2List.length >= 2) {
  var n = s2List.length;
  var t0 = s2List[0].millis;
  var xs = s2List.map(function (p) { return (p.millis - t0) / 86400000; }); // days since first obs
  var ys = s2List.map(function (p) { return p.nbr; });

  // least-squares slope (NBR per day)
  var xMean = xs.reduce(function (a, b) { return a + b; }, 0) / n;
  var yMean = ys.reduce(function (a, b) { return a + b; }, 0) / n;
  var num = 0, den = 0;
  for (var i = 0; i < n; i++) { num += (xs[i] - xMean) * (ys[i] - yMean); den += (xs[i] - xMean) * (xs[i] - xMean); }
  trend = den !== 0 ? num / den : null;

  // wobble = sample std dev of NBR in the window
  var variance = ys.reduce(function (a, b) { return a + Math.pow(b - yMean, 2); }, 0) / (n - 1);
  wobble = Math.sqrt(variance);

  // cycle_count (PROVISIONAL, see file header) = median-crossings in chronological order
  var sorted = ys.slice().sort(function (a, b) { return a - b; });
  var median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  cycleCount = 0;
  for (var j = 1; j < n; j++) {
    var prevSide = ys[j - 1] >= median;
    var curSide = ys[j] >= median;
    if (prevSide !== curSide) cycleCount++;
  }
} else {
  print('STAGE 2 - fewer than 2 clear obs, trend/wobble/cycle_count left null');
}

print('STAGE 2 FEATURES', {
  event_id: EVENT_ID, window_months: T2B_MONTHS, n_obs: s2Count,
  trend_nbr_per_day: trend, wobble_std: wobble, cycle_count_provisional: cycleCount
});

// ---------------------------------------------------------------------------
// STEP 6 - reminder, not enforced by code: this script produces FEATURES only.
// Applying the D6 hand-set thresholds to turn these into a class prediction,
// and comparing that prediction to the already-recorded label with the E2
// structural-vs-trajectory-only split, is a separate, not-yet-written step.
// ---------------------------------------------------------------------------
