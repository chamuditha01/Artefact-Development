// ============================================================================
// event_timeseries.js  -  v6  -  2026-09-12
// Temporal-evidence tool. One event at a time. NBR trajectory for reading
// t_forest and t_loss. Not a classifier/detector, no [T#] values.
//
// FIX FROM v5: valid_pct hit 200 on two-tile dates. Cause: n_total built from
// ee.Image.constant(1) on its own grid, misaligned to the NBR grid n_valid
// counts on. Now n_total = unmasked NBR-band count (same grid), and the
// same-date combine takes max(n_total) not sum. valid_pct now in 0-100.
//
// 2026-10-08: UNCHANGED from v6 except the two lines marked FRESH below
// (EVENT_ID and HANSEN_YEAR). Same instrument as the REF hand-dating, on purpose.
// Fresh test events and their Hansen years (hint only, never t_loss):
//   REF_059 2019 | REF_060 2023 | REF_061 2022 | REF_062 2019
//   REF_063 2023 | REF_064 2024 | REF_065 2023 | REF_066 2024
//   REF_067 2023 | REF_068 2024 | REF_069 2022 | REF_070 2019
// Do hand-dating BEFORE looking at any rule output
// (v1.1/date_test_preregistration.md).
// ============================================================================

// ### SET PER EVENT - update ALL of these each time ###
var SAMPLE_ASSET = 'projects/gsignin-501704/assets/REF_screening_queue';
var PICK_BY      = 'by_id';        // 'by_id' | 'by_area'
var EVENT_ID     = 'REF_059';      // queue_id   <- FRESH: change per event
var EXPECTED_HA  = null;           // set per event OR eyeball printed area vs CSV
var HANSEN_YEAR  = 2019;           // a HINT. NOT t_loss.   <- FRESH: change per event
var MIN_AREA_HA  = 100;            // only for PICK_BY = 'by_area'
var INPUT_MODE   = 'asset';        // 'asset' | 'paste' | 'drawn'

var PASTED_TYPE   = 'MultiPolygon';
var PASTED_COORDS = [[[[80.487,7.838],[80.489,7.838],[80.489,7.840],[80.487,7.840],[80.487,7.838]]]];

var SITE            = 'reference_set';
var SAMPLING_ORIGIN = 'gfw_seeded';

// STEP 1 - period + parameters
var START_DATE = '2019-01-01';
var END_DATE   = '2025-01-01';     // exclusive, includes 2024-12-31
var MIN_VALID_PCT = 60;            // pilot floor, not calibrated
var GUIDE_NBR = 0.40;              // reading aid only

// STEP 2 - functions
function maskClouds(img) {
  var scl = img.select('SCL');
  var keep = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(keep);
}
function addNBR(img) {
  return img.addBands(img.normalizedDifference(['B8','B12']).rename('NBR'));
}

// STEP 3 - resolve geometry
var event;
if (INPUT_MODE === 'asset') {
  var coll = ee.FeatureCollection(SAMPLE_ASSET);
  var feat = PICK_BY === 'by_area'
    ? ee.Feature(coll.filter(ee.Filter.gt('area_ha', MIN_AREA_HA)).first())
    : ee.Feature(coll.filter(ee.Filter.eq('queue_id', EVENT_ID)).first());
  event = feat.geometry();
  print('STEP 3 - loaded feature', feat.toDictionary());
} else if (INPUT_MODE === 'paste') {
  event = PASTED_TYPE === 'MultiPolygon'
    ? ee.Geometry.MultiPolygon(PASTED_COORDS) : ee.Geometry.Polygon(PASTED_COORDS);
} else { event = ee.Geometry(eventGeom); }

var areaHa = event.area({maxError: 1}).divide(10000);
print('STEP 3 - event area (ha)', areaHa);
if (EXPECTED_HA !== null)
  print('STEP 3 - expected (ha)', EXPECTED_HA, '| ratio (want ~1.0)', areaHa.divide(EXPECTED_HA));
print('STEP 3 - geometry parts (1 = single patch)', event.geometries().size());
print('STEP 3 - geometry type', event.type());
Map.centerObject(event, 15);
Map.addLayer(event, {color: 'yellow'}, 'Event ' + EVENT_ID, true);

// STEP 4 - per-image stats
function stats(img) {
  var nbr = img.select('NBR');
  var nValid = nbr.reduceRegion({reducer: ee.Reducer.count(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  // FIX: n_total from UNMASKED NBR band - same grid as n_valid
  var nTotal = nbr.unmask(-99).reduceRegion({reducer: ee.Reducer.count(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  var meanNBR = nbr.reduceRegion({reducer: ee.Reducer.mean(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  var medNBR = nbr.reduceRegion({reducer: ee.Reducer.median(), geometry: event, scale: 10, maxPixels: 1e9}).get('NBR');
  nValid = ee.Number(ee.Algorithms.If(nValid, nValid, 0));
  nTotal = ee.Number(ee.Algorithms.If(nTotal, nTotal, 1));
  return ee.Feature(null, {
    date: img.date().format('YYYY-MM-dd'), NBR: meanNBR, NBR_median: medNBR,
    n_valid: nValid, n_total: nTotal, scene_cloud: img.get('CLOUDY_PIXEL_PERCENTAGE')
  });
}
var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(event).filterDate(START_DATE, END_DATE).map(maskClouds).map(addNBR);
var perImage = ee.FeatureCollection(col.map(stats)).filter(ee.Filter.notNull(['NBR']));

// STEP 5 - combine same-date tiles. n_valid summed, n_total MAX, NBR weighted.
var weighted = perImage.map(function (f) {
  var n = ee.Number(f.get('n_valid'));
  return f.set({w_nbr: ee.Number(f.get('NBR')).multiply(n),
                w_med: ee.Number(f.get('NBR_median')).multiply(n)});
});
var dateList = ee.List(weighted.aggregate_array('date')).distinct();
var dateFC = ee.FeatureCollection(dateList.map(function (d) { return ee.Feature(null, {date: d}); }));
var joined = ee.Join.saveAll('tiles').apply({
  primary: dateFC, secondary: weighted,
  condition: ee.Filter.equals({leftField: 'date', rightField: 'date'})
});
var daily = ee.FeatureCollection(joined.map(function (f) {
  var tiles = ee.FeatureCollection(ee.List(f.get('tiles')));
  var nV = ee.Number(tiles.aggregate_sum('n_valid'));
  var safe = ee.Number(ee.Algorithms.If(nV.gt(0), nV, 1));
  var nT = ee.Number(tiles.aggregate_max('n_total'));   // MAX, not sum
  var d = ee.String(f.get('date'));
  return ee.Feature(null, {
    event_id: EVENT_ID, site: SITE, date: d, millis: ee.Date(d).millis(),
    NBR: ee.Number(tiles.aggregate_sum('w_nbr')).divide(safe),
    NBR_median: ee.Number(tiles.aggregate_sum('w_med')).divide(safe),
    n_valid: nV, n_total: nT, valid_pct: nV.divide(nT).multiply(100),
    n_tiles: tiles.size(), scene_cloud: tiles.aggregate_mean('scene_cloud'),
    hansen_year: HANSEN_YEAR, sampling_origin: SAMPLING_ORIGIN
  });
}));
var clearObs = daily.filter(ee.Filter.gte('valid_pct', MIN_VALID_PCT)).sort('millis');
print('STEP 5 - images in period', perImage.size());
print('STEP 5 - distinct dates', daily.size());
print('STEP 5 - clear dates (valid_pct >= ' + MIN_VALID_PCT + ')', clearObs.size());
print('STEP 5 - dates from >1 tile', daily.filter(ee.Filter.gt('n_tiles', 1)).size());
print('STEP 5 - dates with valid_pct > 100 (MUST be 0)', daily.filter(ee.Filter.gt('valid_pct', 100)).size());
print(ui.Chart.feature.histogram(clearObs, 'valid_pct', 20).setOptions({title: 'valid_pct per date'}));

// STEP 6 - chart
print(ui.Chart.feature.byFeature(clearObs, 'date', ['NBR', 'NBR_median'])
  .setChartType('LineChart')
  .setOptions({
    title: EVENT_ID + ' - NBR trajectory 2019-2024',
    hAxis: {title: 'date', slantedText: true, slantedTextAngle: 60},
    vAxis: {title: 'NBR', viewWindow: {min: -0.2, max: 1.0}, gridlines: {count: 7}},
    pointSize: 4, lineWidth: 1,
    // gaps stay gaps - never join across a blackout
    interpolateNulls: false,
    series: {0: {color: '1f77b4'}, 1: {color: 'aaaaaa', lineDashStyle: [3, 3]}}
  }));
print('Reading guide: dashed grey = median. Mean/median gap = few pixels pulling the average. Guide NBR:', GUIDE_NBR);

// STEP 7 - read t_forest / t_loss by eye. Sustained drop, not a single dip.
// interval = gap between last-forest and first-loss observation. DO NOT interpolate.
// Hansen year is a HINT, never t_loss. Google Earth Step 0 FIRST.
print('STEP 7 - first clear date', ee.Date(clearObs.aggregate_min('millis')).format('YYYY-MM-dd'));
print('STEP 7 - last clear date', ee.Date(clearObs.aggregate_max('millis')).format('YYYY-MM-dd'));

// STEP 8 - exports
var COLS = ['event_id','site','date','NBR','NBR_median','n_valid','n_total',
            'valid_pct','n_tiles','scene_cloud','hansen_year','sampling_origin'];
Export.table.toDrive({collection: clearObs, description: 'ts_' + EVENT_ID + '_clear', fileFormat: 'CSV', selectors: COLS});
Export.table.toDrive({collection: daily.sort('millis'), description: 'ts_' + EVENT_ID + '_all', fileFormat: 'CSV', selectors: COLS});
