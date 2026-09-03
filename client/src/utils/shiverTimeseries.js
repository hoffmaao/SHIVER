/**
 * shiverTimeseries.js
 *
 * Ties the Zarr reader to the numerical pipeline and produces exactly the
 * payload /api/timeseries/multi/json returns, so that MultiSourceMap.vue can
 * consume either source without caring which one answered.
 *
 * This is the browser-side equivalent of get_multi_glacier_timeseries() in
 * server/utils/extract_multi_zarr_ts.py.
 */

import { openRegion, regionForLatitude, resolveSite, readVariable } from './shiverStore.js';
import { processSite, emptySiteResponse } from './timeseriesProcessing.js';
import { npRound } from './timeseriesMath.js';

// The server caps a batch at ten locations and says so in the response.
const MAX_SITES = 10;

/**
 * Extracts timeseries for one or more map locations.
 *
 * Errors are attached to individual sites rather than thrown, matching the
 * server, which always returns a well-formed payload per site so the chart
 * code never has to special-case a partial failure.
 *
 * Because the velocity and error arrays live in separate Zarr chunks, the
 * extraction can report a usable result as soon as the velocity is in and fill
 * the error bars in afterwards. Pass `onPartial` to take advantage of that;
 * without it the function simply resolves once everything is ready.
 *
 * @param {Array<[number, number]>} roi  Locations as [latitude, longitude].
 * @param {object} [settings]
 * @param {number}   [settings.buffer]    Metres around the point, 0 for a single pixel.
 * @param {string[]} [settings.variable]  Variables to extract, e.g. ['speed'].
 * @param {string[]} [settings.sources]   Restrict to these datasets; empty means all.
 * @param {number}   [settings.gap_fill]  Max days of gap to bridge.
 * @param {number}   [settings.win_raw]   Savitzky-Golay window for raw points.
 * @param {number}   [settings.win_daily] Savitzky-Golay window for the daily average.
 * @param {number}   [settings.poly]      Savitzky-Golay polynomial order.
 * @param {object} [hooks]
 * @param {(results: object) => void} [hooks.onPartial] Called with the payload
 *        once velocities are ready but before error bars have been fetched.
 * @param {(progress: object) => void} [hooks.onProgress] Called as sites complete.
 * @returns {Promise<object>} Payload keyed by site name, e.g. { Site_0: {...} }.
 */
export async function extractTimeseries(roi, settings = {}, hooks = {}) {
  const {
    buffer = 500,
    variable = ['speed'],
    sources = [],
    gap_fill: gapFill = 24,
    win_raw: winRaw = 25,
    win_daily: winDaily = 25,
    poly = 2,
  } = settings;

  const results = {};

  // 1. Apply the same ten-location cap the server enforces.
  let locations = roi;
  if (locations.length > MAX_SITES) {
    locations = locations.slice(0, MAX_SITES);
    results.warning = 'File contained more than 10 locations. Only the first 10 were extracted.';
  }

  const processOptions = { variable, sources, gapFill, winRaw, winDaily, poly };

  // 2. Read every site's velocities first. Errors come in a second pass so a
  //    chart can be drawn from roughly half the bytes.
  const pending = [];

  for (let index = 0; index < locations.length; index++) {
    const [latitude, longitude] = locations[index];
    const siteName = `Site_${index}`;

    const meta = buildMeta(siteName, latitude, longitude, buffer, variable, sources, {
      gap: gapFill, win_raw: winRaw, win_daily: winDaily, poly,
    });

    try {
      const regionInfo = await openRegion(meta.region);
      const site = resolveSite(regionInfo, latitude, longitude, buffer);

      if (site.error) {
        results[siteName] = { ...emptySiteResponse('error', site.error, variable), meta };
        continue;
      }

      // 3. Velocities for every requested variable.
      const values = {};
      const errors = {};
      let validCount = null;

      for (const name of variable) {
        const read = await readVariable(regionInfo, name, site);
        values[name] = read.values;
        // The server derives the pixel count from the first variable only.
        if (validCount === null) validCount = read.validCount;
        errors[name] = new Float64Array(regionInfo.times.length).fill(NaN);
      }

      const siteInput = {
        times: regionInfo.times,
        timeSeparation: regionInfo.timeSeparation,
        dataSource: regionInfo.dataSource,
        validCount,
        values,
        errors,
      };

      results[siteName] = { ...processSite(siteInput, processOptions), meta };
      pending.push({ siteName, regionInfo, site, siteInput, meta });
    } catch (error) {
      // Reaching here means the store itself could not be read - the CDN is
      // down, CORS broke, or a chunk failed to decode. That affects every site
      // equally, so we surface it to the caller rather than marking one site
      // bad, and the caller can fall back to the server.
      throw new StoreUnavailableError(error);
    }

    hooks.onProgress?.({ completed: index + 1, total: locations.length, stage: 'velocity' });
  }

  // 4. Hand back a drawable result before spending bytes on the error bars.
  if (hooks.onPartial && pending.length > 0) {
    hooks.onPartial(results);
  }

  // 5. Second pass: uncertainties.
  for (let index = 0; index < pending.length; index++) {
    const { siteName, regionInfo, site, siteInput, meta } = pending[index];

    try {
      for (const name of variable) {
        const read = await readVariable(regionInfo, `${name}_error`, site);
        siteInput.errors[name] = read.values;
      }
      results[siteName] = { ...processSite(siteInput, processOptions), meta };
    } catch (error) {
      // Uncertainties are a nice-to-have: keep the velocities we already have
      // rather than failing the whole site over missing error bars.
      console.warn(`SHIVER: error bars unavailable for ${siteName}:`, error);
    }

    hooks.onProgress?.({ completed: index + 1, total: pending.length, stage: 'error' });
  }

  return results;
}

/**
 * Builds the meta block the server attaches to every site. MultiSourceMap.vue
 * reads site_name, lat, lon and buffer_used out of this for its labels,
 * exports and CSV headers, so the shape has to match.
 */
function buildMeta(siteName, latitude, longitude, buffer, variable, sources, params) {
  return {
    site_name: siteName,
    region: regionForLatitude(latitude),
    buffer_used: buffer,
    lat: npRound(latitude, 5),
    lon: npRound(longitude, 5),
    // The map only ever sends points; polygons still arrive by file upload,
    // which continues to go to the server.
    type: 'Point',
    variable,
    sources_requested: sources && sources.length > 0 ? sources : 'All',
    params,
  };
}

/**
 * Raised when the Zarr store cannot be read at all, as distinct from a site
 * that legitimately has no data. Callers use this to decide whether falling
 * back to the FastAPI backend is worth trying.
 */
export class StoreUnavailableError extends Error {
  constructor(cause) {
    super(`Could not read the SHIVER data store: ${cause?.message || String(cause)}`);
    this.name = 'StoreUnavailableError';
    this.cause = cause;
  }
}
