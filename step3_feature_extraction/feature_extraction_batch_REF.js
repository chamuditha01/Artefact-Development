// ============================================================================
// feature_extraction_batch_REF.js  -  v5  -  2026-09-28
// Batch Stage 1 / Stage 2 feature extraction for the labelled REF events.
// Run in GROUPS OF 5. See feature_extraction_v1.js for full documentation.
//
// TO RUN: set GROUP below, click Run, then go to the Tasks tab and click Run
// on the export task that appears. Repeat for GROUP = 0, 1, 2, 3, 4, 5.
// ============================================================================

var SAMPLE_ASSET  = 'projects/gsignin-501704/assets/REF_screening_queue';
var MIN_VALID_PCT = 60;
var T1_MONTHS = 12, T2A_DAYS = 60, T2B_MONTHS = 12;

var HANSEN_ID         = 'UMD/hansen/global_forest_change_2025_v1_13';
var HANSEN_LOSSYEAR   = ee.Image(HANSEN_ID).select('lossyear');
var SEARCH_RADIUS_M   = 10000;
var SEARCH_RADIUS_PX  = Math.ceil(SEARCH_RADIUS_M / 30);

// ### SET PER RUN ###
var GROUP = 0;        // 0..5
var GROUP_SIZE = 5;

var REF_EVENTS = [
  ['REF_004','2022-02-04','2022-03-11','cultivation','confirmed'],
  ['REF_039','2023-02-14','2023-03-06','uncertain_logging','confirmed'],
  ['REF_055','2023-06-16','2023-07-16','cultivation','confirmed'],
  ['REF_026','2024-06-30','2024-07-20','uncertain_logging','confirmed'],
  ['REF_058','2023-01-20','2023-03-26','cultivation','candidate'],
  ['REF_053','2024-05-26','2024-05-31','permanent_encroachment','confirmed'],
  ['REF_005','2022-02-01','2022-03-13','uncertain','confirmed'],
  ['REF_028','2023-06-16','2023-08-05','uncertain','confirmed'],
  ['REF_041','2022-07-01','2022-07-26','uncertain','confirmed'],
  ['REF_013','2023-03-06','2023-03-16','permanent_encroachment','confirmed'],
  ['REF_020','2021-06-29','2021-08-23','natural','confirmed'],
  ['REF_008','2020-01-31','2020-02-10','cultivation_pasture','confirmed'],
  ['REF_021','2023-03-01','2023-03-11','cultivation','confirmed'],
  ['REF_016','2022-04-25','2022-04-30','cultivation','confirmed'],
  ['REF_044','2019-11-17','2019-12-27','permanent_encroachment','confirmed'],
  ['REF_047','2022-03-11','2022-03-16','uncertain','confirmed'],
  ['REF_034','2023-06-04','2023-06-24','cultivation','confirmed'],
  ['REF_049','2020-02-20','2020-03-01','permanent_encroachment','confirmed'],
  ['REF_052','2022-07-01','2022-07-26','cultivation','confirmed'],
  ['REF_003','2024-01-22','2024-01-27','uncertain','confirmed'],
  ['REF_018','2023-07-21','2023-08-25','uncertain_logging','confirmed'],
  ['REF_014','2024-08-14','2024-09-18','cultivation','confirmed'],
  ['REF_023','2021-01-30','2021-02-14','cultivation','confirmed'],
  ['REF_012','2021-06-09','2021-08-23','permanent_encroachment','candidate'],
  ['REF_007','2020-02-02','2020-03-18','cultivation','confirmed'],
  ['REF_042','2024-05-14','2024-06-23','uncertain','confirmed'],
  ['REF_033','2022-04-25','2022-04-30','cultivation','confirmed'],
  ['REF_031','2022-06-26','2022-08-10','cultivation','confirmed'],
  ['REF_025','2016-11-01','2017-06-01','permanent_encroachment','visual_estimate']
];

var slice = REF_EVENTS.slice(GROUP * GROUP_SIZE, (GROUP + 1) * GROUP_SIZE);
print('GROUP ' + GROUP + ' - events in this run', slice.map(function (e) { return e[0]; }));
if (slice.length === 0) throw new Error('GROUP ' + GROUP + ' is empty - only groups 0-5 exist for 29 events.');

function maskClouds(img) {
  var scl = img.select('SCL');
  var keep = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(keep);
}
function addNBR(img) {
  return img.addBands(img.normalizedDifference(['B8', 'B12']).rename('NBR'));
}
function statsFor(geom) {
  return function (img) {
    var nbr = img.select('NBR');
    var nValid = nbr.reduceRegion({reducer: ee.Reducer.count(), geometry: geom, scale: 10, maxPixels: 1e9}).get('NBR');
    var nTotal = nbr.unmask(-99).reduceRegion({reducer: ee.Reducer.count(), geometry: geom, scale: 10, maxPixels: 1e9}).get('NBR');
    var meanNBR = nbr.reduceRegion({reducer: ee.Reducer.mean(), geometry: geom, scale: 10, maxPixels: 1e9}).get('NBR');
    nValid = ee.Number(ee.Algorithms.If(nValid, nValid, 0));
    nTotal = ee.Number(ee.Algorithms.If(nTotal, nTotal, 1));
    return ee.Feature(null, {
      millis: img.date().millis(), NBR: meanNBR,
      n_valid: nValid, n_total: nTotal, valid_pct: nValid.divide(nTotal).multiply(100)
    });
  };
}
function clearObsInRange(geom, startDate, endDate) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom).filterDate(startDate, endDate)
    .map(maskClouds).map(addNBR);
  var perImage = ee.FeatureCollection(col.map(statsFor(geom))).filter(ee.Filter.notNull(['NBR']));
  return perImage.filter(ee.Filter.gte('valid_pct', MIN_VALID_PCT)).sort('millis');
}
function outsideWindow(obsFC, startMillis, endMillis) {
  return obsFC.filter(ee.Filter.or(ee.Filter.lt('millis', startMillis), ee.Filter.gt('millis', endMillis))).size();
}

var coll = ee.FeatureCollection(SAMPLE_ASSET);
var results = [];

slice.forEach(function (row) {
  var eventId = row[0], tForestStr = row[1], tLossStr = row[2], knownClass = row[3], dateReliability = row[4];
  var feat = ee.Feature(coll.filter(ee.Filter.eq('queue_id', eventId)).first());
  var geom = feat.geometry();
  var tForest = ee.Date(tForestStr), tLoss = ee.Date(tLossStr);

  var baselineStart = tForest.advance(-T1_MONTHS, 'month');
  var baselineObs = clearObsInRange(geom, baselineStart, tForest);
  var baselineInfo = ee.Dictionary({
    n: baselineObs.size(),
    bad: outsideWindow(baselineObs, baselineStart.millis(), tForest.millis()),
    iPre: baselineObs.aggregate_mean('NBR')
  }).getInfo();
  if (baselineInfo.bad > 0) throw new Error('FIREWALL VIOLATION - BASELINE - ' + eventId);

  var eventYearCode = tLoss.get('year').subtract(2000);
  var priorLossMask = HANSEN_LOSSYEAR.gt(0).and(HANSEN_LOSSYEAR.lt(eventYearCode))
    .rename('lossmask').selfMask();
  var lossDistImg = priorLossMask.fastDistanceTransform(SEARCH_RADIUS_PX).sqrt()
    .multiply(ee.Image.pixelArea().sqrt());
  var lossDistDict = lossDistImg.reduceRegion({
    reducer: ee.Reducer.min(), geometry: geom, scale: 30, maxPixels: 1e9
  });
  var distToPriorLoss = ee.Algorithms.If(lossDistDict.size().gt(0), lossDistDict.values().get(0), null);

  var s1Start = tLoss, s1End = tLoss.advance(T2A_DAYS, 'day');
  var s1Obs = clearObsInRange(geom, s1Start, s1End);
  var perimeter = geom.perimeter({maxError: 1}), area = geom.area({maxError: 1});
  var compactness = ee.Number(4).multiply(Math.PI).multiply(area).divide(perimeter.pow(2));
  var meanSlope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003'))
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: geom, scale: 30, maxPixels: 1e9}).get('slope');
  var s1Info = ee.Dictionary({
    n: s1Obs.size(),
    bad: outsideWindow(s1Obs, s1Start.millis(), s1End.millis()),
    minNBR: s1Obs.aggregate_min('NBR'), compactness: compactness, slope: meanSlope,
    area_ha: area.divide(10000), distToPriorLoss: distToPriorLoss
  }).getInfo();
  if (s1Info.bad > 0) throw new Error('FIREWALL VIOLATION - STAGE 1 - ' + eventId);
  var earlySpectralChange = (s1Info.n > 0 && s1Info.minNBR !== null && baselineInfo.iPre !== null)
    ? (baselineInfo.iPre - s1Info.minNBR) : null;

  var s2Start = tLoss, s2End = tLoss.advance(T2B_MONTHS, 'month');
  var s2Obs = clearObsInRange(geom, s2Start, s2End);
  var s2BadCount = outsideWindow(s2Obs, s2Start.millis(), s2End.millis()).getInfo();
  if (s2BadCount > 0) throw new Error('FIREWALL VIOLATION - STAGE 2 - ' + eventId);
  var s2Raw = s2Obs.select(['millis', 'NBR']).getInfo();
  var s2List = s2Raw.features.map(function (f) { return {millis: f.properties.millis, nbr: f.properties.NBR}; });

  var trend = null, wobble = null, cycleCount = null, n2 = s2List.length;
  if (n2 >= 2) {
    var t0 = s2List[0].millis;
    var xs = s2List.map(function (p) { return (p.millis - t0) / 86400000; });
    var ys = s2List.map(function (p) { return p.nbr; });
    var xMean = xs.reduce(function (a, b) { return a + b; }, 0) / n2;
    var yMean = ys.reduce(function (a, b) { return a + b; }, 0) / n2;
    var num = 0, den = 0;
    for (var i = 0; i < n2; i++) { num += (xs[i] - xMean) * (ys[i] - yMean); den += (xs[i] - xMean) * (xs[i] - xMean); }
    trend = den !== 0 ? num / den : null;
    var variance = ys.reduce(function (a, b) { return a + Math.pow(b - yMean, 2); }, 0) / (n2 - 1);
    wobble = Math.sqrt(variance);
    var sorted = ys.slice().sort(function (a, b) { return a - b; });
    var median = n2 % 2 ? sorted[(n2 - 1) / 2] : (sorted[n2 / 2 - 1] + sorted[n2 / 2]) / 2;
    cycleCount = 0;
    for (var j = 1; j < n2; j++) { if ((ys[j - 1] >= median) !== (ys[j] >= median)) cycleCount++; }
  }

  var row_out = {
    event_id: eventId, known_class: knownClass, date_reliability: dateReliability,
    area_ha: s1Info.area_ha, i_pre: baselineInfo.iPre, n_obs_baseline: baselineInfo.n,
    shape_compactness: s1Info.compactness, slope_deg: s1Info.slope, dist_to_prior_loss_m: s1Info.distToPriorLoss,
    n_obs_s1: s1Info.n, early_spectral_change: earlySpectralChange,
    n_obs_s2: n2, trend_nbr_per_day: trend, wobble_std: wobble, cycle_count_provisional: cycleCount
  };
  results.push(row_out);
  print(eventId + ' done', row_out);
});

var COLS = ['event_id', 'known_class', 'date_reliability', 'area_ha', 'i_pre', 'n_obs_baseline',
  'shape_compactness', 'slope_deg', 'dist_to_prior_loss_m', 'n_obs_s1', 'early_spectral_change',
  'n_obs_s2', 'trend_nbr_per_day', 'wobble_std', 'cycle_count_provisional'];
var outFC = ee.FeatureCollection(results.map(function (r) { return ee.Feature(null, r); }));
Export.table.toDrive({ collection: outFC, description: 'features_REF_group' + GROUP, fileFormat: 'CSV', selectors: COLS });
print('GROUP ' + GROUP + ' - export queued: features_REF_group' + GROUP + ' (go to Tasks tab and click Run)');
