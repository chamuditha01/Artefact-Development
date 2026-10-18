// ============================================================================
// REF_fresh_timeseries_export.js  -  v1  -  2026-10-07
// Same method as REF_batch_timeseries_export.js v3 (count in its own
// reduceRegion call, mosaic by day, NO erosion). Only the events and the export
// name change. FRESH test events REF_059-REF_070, for the date-rule test
// (v1.1/date_test_preregistration.md). RUN IN GROUPS: change GROUP to 7, 8, 9,
// click Run, then press Run on the export task in the Tasks tab each time.
// Do NOT print allSeries.size() (memory limit, B22).
// ============================================================================

var SAMPLE_ASSET = 'projects/gsignin-501704/assets/REF_screening_queue';
var ID_FIELD     = 'queue_id';

var GROUP = 7;   // 7, 8 or 9
var GROUPS = {
  7: ['REF_059','REF_060','REF_061','REF_062','REF_063'],
  8: ['REF_064','REF_065','REF_066','REF_067','REF_068'],
  9: ['REF_069','REF_070']
};
var EVENTS = GROUPS[GROUP];

var START_DATE = '2019-01-01';
var END_DATE   = '2026-09-25';     // same as the REF export, so the series are comparable
var MIN_VALID_PCT = 60;            // exported as a column, NOT applied as a filter

var queue = ee.FeatureCollection(SAMPLE_ASSET)
              .filter(ee.Filter.inList(ID_FIELD, EVENTS)).sort(ID_FIELD);
print('GROUP ' + GROUP + ' - events (want ' + EVENTS.length + ')', queue.aggregate_array(ID_FIELD));

function maskClouds(img) {
  var scl = img.select('SCL');
  var keep = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(keep);
}
function addNBR(img) {
  return img.addBands(img.normalizedDifference(['B8','B12']).rename('NBR'));
}

var STAT_REDUCER = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.median(), sharedInputs: true});

function eventSeries(f) {
  f = ee.Feature(f);
  var id = f.get(ID_FIELD), site = f.get('site'), hy = f.get('hansen_lossyear');
  var geom = f.geometry();   // NOT eroded

  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom).filterDate(START_DATE, END_DATE)
    .map(maskClouds).map(addNBR).select('NBR');

  var days = ee.List(col.aggregate_array('system:time_start'))
    .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }).distinct().sort();

  return ee.FeatureCollection(days.map(function (d) {
    d = ee.String(d);
    var t0 = ee.Date(d);
    var sub = col.filterDate(t0, t0.advance(1, 'day'));

    var proj = ee.Image(sub.first()).select('NBR').projection();
    var nbr = ee.Image(sub.mosaic()).setDefaultProjection(proj);

    var s = nbr.reduceRegion({reducer: STAT_REDUCER, geometry: geom,
                              scale: 10, maxPixels: 1e9});
    var nTotal = ee.Number(nbr.unmask(-99).reduceRegion({
      reducer: ee.Reducer.count(), geometry: geom,
      scale: 10, maxPixels: 1e9}).get('NBR'));
    nTotal = ee.Number(ee.Algorithms.If(nTotal, nTotal, 1));
    // count is its own separate reduceRegion() call (B23 fix)
    var nValidRaw = nbr.reduceRegion({reducer: ee.Reducer.count(), geometry: geom,
                                       scale: 10, maxPixels: 1e9}).get('NBR');
    var nValid = ee.Number(ee.Algorithms.If(nValidRaw, nValidRaw, 0));

    return ee.Feature(null, {
      event_id: id, site: site, hansen_year: hy,
      date: d, year: t0.get('year'),
      NBR: s.get('NBR_mean'), NBR_median: s.get('NBR_median'),
      n_valid: nValid, n_total: nTotal,
      valid_pct: nValid.divide(nTotal).multiply(100),
      n_tiles: sub.size(),
      scene_cloud: sub.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE'),
      min_valid_pct_setting: MIN_VALID_PCT
    });
  })).filter(ee.Filter.notNull(['NBR']));
}

var allSeries = ee.FeatureCollection(
  queue.toList(queue.size()).map(eventSeries)).flatten();

Map.addLayer(queue, {color: 'red'}, 'group ' + GROUP, true);
Map.centerObject(queue, 9);

Export.table.toDrive({
  collection: allSeries.sort('event_id'),
  description: 'REF_fresh_timeseries_g' + GROUP,
  fileFormat: 'CSV',
  selectors: ['event_id','site','hansen_year','date','year','NBR','NBR_median',
              'n_valid','n_total','valid_pct','n_tiles','scene_cloud',
              'min_valid_pct_setting']
});
