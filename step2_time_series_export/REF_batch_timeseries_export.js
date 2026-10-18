// ============================================================================
// REF_batch_timeseries_export.js  -  v2  -  2026-09-24
// Batch NBR time-series export for the 30 confirmed REF events.
//
// Matches how the CALIBRATION batch was actually processed
// (cal_batch_timeseries_v72.js), not the earlier draft of this script:
//   1. NO core-pixel erosion. The frozen protocol (Sec 4 step 1) says the
//      polygon should be eroded by [T5]=10m before measuring. The actual
//      script that produced calibration_batch.csv never did this - erosion
//      was only ever a separate DIAGNOSTIC (core pixel counts at 0/10/20/30m,
//      used to pick the [T5] value), never applied to the real trajectory
//      data. To keep REF measured the same way as CAL (so thresholds set
//      from CAL still apply), this script also does NOT erode. Logged as a
//      correction in decisions.md, 2026-09-24 - the protocol text is being
//      fixed to match what was actually done, not the other way round.
//   2. Mosaic-first, combined reducer (mean+median+count in one pass) -
//      avoids the B14/B15 bugs, same fix as v7.2.
//   3. RUN IN GROUPS OF FIVE - 30 events in one go blew GEE's memory limit
//      (same thing happened on the CAL batch). Change GROUP below and export
//      six times.
//   4. END_DATE extended to read the archive up to today.
//
// Output columns match CAL_all_timeseries.csv exactly:
//   event_id, site, hansen_year, date, year, NBR, NBR_median,
//   n_valid, n_total, valid_pct, n_tiles, scene_cloud, min_valid_pct_setting
// (n_images from cal_batch_timeseries_v72.js is exported here as n_tiles to
//  match the CAL_all_timeseries.csv header exactly - same value.)
// ============================================================================

var SAMPLE_ASSET = 'projects/gsignin-501704/assets/REF_screening_queue';
var ID_FIELD     = 'queue_id';
// NOTE: 'queue_id' is unverified against the asset's real column names -
// confirm if GROUP below prints 0 events.

// *** RUN IN GROUPS OF FIVE. Change GROUP to 1-6 and export each time. ***
var GROUP = 1;
var GROUPS = {
  1: ['REF_003','REF_004','REF_005','REF_007','REF_008'],
  2: ['REF_012','REF_013','REF_014','REF_016','REF_018'],
  3: ['REF_020','REF_021','REF_023','REF_025','REF_026'],
  4: ['REF_028','REF_031','REF_033','REF_034','REF_039'],
  5: ['REF_040','REF_041','REF_042','REF_044','REF_047'],
  6: ['REF_049','REF_052','REF_053','REF_055','REF_058']
};
var EVENTS = GROUPS[GROUP];

var START_DATE = '2019-01-01';
var END_DATE   = '2026-09-25';     // read to date, not just study-period end
var MIN_VALID_PCT = 60;            // exported as a column, NOT applied as a filter

// ============================================================================

var queue = ee.FeatureCollection(SAMPLE_ASSET)
              .filter(ee.Filter.inList(ID_FIELD, EVENTS)).sort(ID_FIELD);
print('GROUP ' + GROUP + ' - events (want 5)', queue.aggregate_array(ID_FIELD));

function maskClouds(img) {
  var scl = img.select('SCL');
  var keep = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  return img.updateMask(keep);
}
function addNBR(img) {
  return img.addBands(img.normalizedDifference(['B8','B12']).rename('NBR'));
}

// Combined reducer - mean, median and count in ONE pass over the pixels.
var STAT_REDUCER = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.count(),  sharedInputs: true});

function eventSeries(f) {
  f = ee.Feature(f);
  var id = f.get(ID_FIELD), site = f.get('site'), hy = f.get('hansen_lossyear');
  var geom = f.geometry();   // NOT eroded - matches how CAL was measured

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
    var nValid = ee.Number(ee.Algorithms.If(s.get('NBR_count'), s.get('NBR_count'), 0));

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
  description: 'REF_timeseries_v1_g' + GROUP,
  fileFormat: 'CSV',
  selectors: ['event_id','site','hansen_year','date','year','NBR','NBR_median',
              'n_valid','n_total','valid_pct','n_tiles','scene_cloud',
              'min_valid_pct_setting']
});

// ---------------------------------------------------------------------------
// NOTES
// ---------------------------------------------------------------------------
// - Do NOT print allSeries.size() interactively - that is what blew the
//   memory limit last time. Trust the export task; check its row count from
//   the downloaded CSV instead.
// - Run all 6 groups (change GROUP each time), then merge the 6 downloaded
//   CSVs into one REF_all_timeseries.csv (same header, just stack the rows -
//   send them to me and I'll merge them).
// ---------------------------------------------------------------------------
