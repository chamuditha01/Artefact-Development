// ============================================================================
// assign_climatic_zone.js  -  2026-09-29 (v5 - the actual fix, confirmed by
// debug_precip_null.js, not a guess)
// Assigns each Sri Lankan WDPA polygon to one of three climatic zones (dry /
// intermediate / wet), per the Department of Meteorology's rainfall-based
// definition (sources.md, 2026-09-29, Mendis presentation):
//   Dry zone         : R < 1750mm
//   Intermediate zone: 1750mm <= R < 2500mm
//   Wet zone         : R >= 2500mm
// where R = mean annual precipitation.
//
// No public GIS boundary layer exists for these 3 zones (DoA's map is a JPG,
// not a shapefile - see sources.md notes), so the zone is computed directly
// from a rainfall climatology raster against the stated thresholds, rather
// than digitized from a scanned map. This feeds D4 (evaluation stratification
// rule) - own scoping decision (2026-09-29, supervisor delegated): 3-zone
// system chosen over the finer agro-ecological-region system, to avoid
// fragmenting already-thin per-class event counts (M20/B3) given the
// deadline. NOT a decision about detection scope - detection is already
// national (project.md); this only decides how evaluation is stratified.
//
// CHIRPS climatology period: 1991-2020 (WMO 30-year climate normal), NOT the
// 2019-2024 study period - a 6-year window is weather, not climate.
//
// DEBUG HISTORY (2026-09-29), for the record - do not repeat these attempts:
// v1: crashed, no null guard on STEP 4's classification.
// v2: added null guard + `bestEffort` on reduceRegions - invalid argument
//     there (only valid on singular reduceRegion), threw a hard error.
// v3: removed bestEffort; GUESSED the null came from invalid WDPA geometry
//     and added a buffer(0,1) "repair" step applied to all 679 polygons.
//     Wrong guess, and it broke every polygon (679/679 no_data).
// v4: GUESSED the null came from a heavy chained year-by-year sum/mean
//     climatology (real bug pattern this project hit before, issues.md B15)
//     and replaced it with a single flat mean() * 365.25. Still 679/679
//     no_data - because the real bug (v3's buffer) was still in the script.
// CONFIRMED FIX: ran debug_precip_null.js, isolating each step on Wilpattu
// alone. reduceRegion (singular) on Wilpattu's RAW, unbuffered geometry
// returned a sane value (1295.3mm - well inside Sri Lanka's real ~900-5000mm
// range). The only difference between that working call and the broken
// pipeline was the buffer(0,1) "repair" from v3, applied to every polygon
// before reducing. That was never a real bug - it was an untested guess
// that introduced one. Removed entirely in v5. No geometry repair is
// applied; if a genuine geometry-validity no_data case turns up in this
// run, it will show in STEP 4b and can be investigated on its own evidence,
// not fixed pre-emptively again.
// ============================================================================

var CLIMATOLOGY_START = '1991-01-01';
var CLIMATOLOGY_END   = '2021-01-01';  // exclusive end -> 30 full years
var DRY_MAX_MM   = 1750;
var WET_MIN_MM   = 2500;

// STEP 1 - all Sri Lankan WDPA polygons, RAW - no geometry transform of any
// kind. (Still does not apply D11's designation filter - re-filter
// downstream once D11 is finalised.)
var wdpa = ee.FeatureCollection('WCMC/WDPA/current/polygons')
             .filter(ee.Filter.eq('ISO3', 'LKA'));
print('STEP 1 - WDPA polygons (LKA, unfiltered by designation/status)', wdpa.size());

// STEP 2 - mean annual precipitation, 1991-2020: one flat mean() over every
// daily CHIRPS image in the window, scaled by 365.25. Confirmed working
// (debug script) - not the year-by-year chained version from v1-v2.
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
               .filterDate(CLIMATOLOGY_START, CLIMATOLOGY_END)
               .select('precipitation');
var meanAnnualPrecip = chirps.mean().multiply(365.25)
                          .rename('mean_annual_precip_mm');

// STEP 3 - mean annual precipitation per polygon, on the RAW wdpa collection.
var withPrecip = meanAnnualPrecip.reduceRegions({
  collection: wdpa,
  reducer: ee.Reducer.mean(),
  scale: 5566,  // native CHIRPS resolution (~0.05deg)
  tileScale: 8
});

// STEP 4 - classify by the stated thresholds. Still null-safe, in case a
// genuine (not manufactured) no_data case exists.
var zoned = withPrecip.map(function (f) {
  var r = f.get('mean_annual_precip_mm');
  var zone = ee.Algorithms.If(
    ee.Algorithms.IsEqual(r, null),
    'no_data',
    ee.Algorithms.If(
      ee.Number(r).lt(DRY_MAX_MM), 'dry',
      ee.Algorithms.If(ee.Number(r).lt(WET_MIN_MM), 'intermediate', 'wet')
    )
  );
  return f.set('climatic_zone', zone);
});

print('STEP 4 - polygons by zone', zoned.aggregate_histogram('climatic_zone'));

// STEP 4b - list any no_data cases by name/area - now a real finding if it
// shows anything, not a manufactured one.
var noData = zoned.filter(ee.Filter.eq('climatic_zone', 'no_data'));
print('STEP 4b - no_data polygons (name, GIS_AREA, REP_AREA) - want an empty list',
  noData.size(), noData.select(['NAME', 'GIS_AREA', 'REP_AREA']));

// STEP 4c - spot check raw values on 5 polygons.
print('STEP 4c - spot check: 5 polygons with their raw precip value',
  zoned.limit(5).select(['NAME', 'mean_annual_precip_mm', 'climatic_zone']));

// STEP 5 - sanity check against the two sites already assumed in the
// existing framing (Wilpattu=dry, Sinharaja=wet).
var checkNames = ['Wilpattu', 'Sinharaja'];
print('STEP 5 - sanity check (expect Wilpattu=dry ~1295mm per debug run, Sinharaja=wet)',
  zoned.filter(ee.Filter.inList('NAME', checkNames))
       .select(['NAME', 'DESIG_ENG', 'mean_annual_precip_mm', 'climatic_zone']));

// STEP 6 - export.
var CSV_COLS = ['NAME', 'DESIG_ENG', 'STATUS', 'REP_AREA', 'GIS_AREA',
                'mean_annual_precip_mm', 'climatic_zone'];
Export.table.toDrive({
  collection: zoned, description: 'WDPA_LKA_climatic_zone',
  fileFormat: 'CSV', selectors: CSV_COLS
});
Export.table.toAsset({
  collection: zoned, description: 'WDPA_LKA_climatic_zone',
  assetId: 'WDPA_LKA_climatic_zone'
});

// RUN THIS. Expect STEP 4b empty or near-empty, STEP 4c showing values in
// roughly the 900-5000mm range, and STEP 5 matching Wilpattu=dry
// (~1295mm, confirmed by debug_precip_null.js), Sinharaja=wet. If Sinharaja
// does NOT come back wet, STOP - that's a real finding about the existing
// E7 framing, not a script bug.
