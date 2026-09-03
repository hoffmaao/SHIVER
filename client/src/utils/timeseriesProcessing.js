/**
 * timeseriesProcessing.js
 *
 * Browser port of _process_single_site_multi() from
 * server/utils/extract_multi_zarr_ts.py.
 *
 * This module deliberately knows nothing about Zarr, HTTP or the map. It takes
 * arrays that have already been pulled off the time axis for one site and
 * returns exactly the JSON payload shape that /api/timeseries/multi/json
 * returns, so MultiSourceMap.vue cannot tell which side produced it.
 *
 * The processing is, in order:
 *   1. filter by data source, 2. tidy the time separations, 3. build the
 *   combined exact + daily timeline, 4. reject outliers, 5. smooth the raw
 *   points, 6. build a duration-weighted daily average, 7. smooth that too.
 *
 * Missing values are NaN internally and null on the way out, matching the
 * server's clean_nans().
 */

import {
  npRound,
  interpolateTime,
  rollingMeanStd,
  nanStd,
  savgolFilter,
  effectiveWindow,
} from './timeseriesMath.js';

const MS_PER_DAY = 86400000;

// The server treats these as physically impossible and drops them before any
// smoothing happens. Speeds are signed magnitudes so a small negative value is
// tolerated; directional components are only bounded in magnitude.
const SPEED_MIN = -100;
const ABSOLUTE_MAX = 100000;

// --- 1. SMALL HELPERS ---

/** Largest whole day at or below a timestamp. */
function floorDay(ms) {
  return Math.floor(ms / MS_PER_DAY) * MS_PER_DAY;
}

/** Smallest whole day at or above a timestamp. */
function ceilDay(ms) {
  return Math.ceil(ms / MS_PER_DAY) * MS_PER_DAY;
}

/**
 * Formats a timestamp as the server does with strftime('%Y-%m-%dT%H:%M:%S').
 * Note the deliberate absence of a timezone suffix - the server's timestamps
 * are naive, and appending a 'Z' here would shift every point in the chart.
 */
function formatTimestamp(ms) {
  return new Date(ms).toISOString().slice(0, 19);
}

/**
 * Reproduces clean_nans(): anything not finite becomes null so it survives
 * JSON, while strings are passed through untouched.
 */
function cleanNaNs(values) {
  return Array.from(values, (v) => {
    if (typeof v === 'string') return v;
    return Number.isFinite(v) ? v : null;
  });
}

/** Rounds a whole series with numpy semantics, preserving gaps. */
function roundSeries(values, decimals) {
  return Array.from(values, (v) => (Number.isFinite(v) ? npRound(v, decimals) : null));
}

/**
 * The standardised empty payload from _empty_site_response(). Returning the
 * full key structure even on failure is what stops the chart code downstream
 * from having to guard every property access.
 */
export function emptySiteResponse(status, message, variables = ['speed']) {
  const data = { dates: [], dt: [], data_source: [], count: [] };

  for (const variable of variables) {
    data[`${variable}_error`] = [];
    data[variable] = { raw: [], smoothed: [] };
  }

  return { status, message, data };
}

// --- 2. TIMELINE CONSTRUCTION ---

/**
 * Builds the combined timeline the server plots against: every observation
 * timestamp, unioned with a complete daily grid spanning them.
 *
 * The daily grid is what gives the smoothed series its regular spacing, while
 * the exact timestamps keep the raw points on their true acquisition dates.
 * Duplicates collapse, exactly as a pandas Index union does.
 */
export function buildFullIndex(observationTimes) {
  if (observationTimes.length === 0) return [];

  const start = floorDay(observationTimes[0]);
  const end = ceilDay(observationTimes[observationTimes.length - 1]);

  const merged = new Set();
  for (let day = start; day <= end; day += MS_PER_DAY) merged.add(day);
  for (const t of observationTimes) merged.add(t);

  return Array.from(merged).sort((a, b) => a - b);
}

/**
 * Scatters observation-indexed values onto the full timeline, leaving gaps
 * wherever the timeline has no matching observation. Equivalent to
 * Series.reindex(full_idx).
 */
function reindexOntoTimeline(values, observationTimes, positionOf, timelineLength, fill = NaN) {
  const out = new Array(timelineLength).fill(fill);

  for (let i = 0; i < observationTimes.length; i++) {
    const pos = positionOf.get(observationTimes[i]);
    if (pos !== undefined) out[pos] = values[i];
  }
  return out;
}

// --- 3. MAIN PIPELINE ---

/**
 * Processes one site's extracted arrays into the API response payload.
 *
 * @param {object} site
 * @param {number[]} site.times            Observation times, ms since epoch.
 * @param {number[]} site.timeSeparation   Days spanned by each acquisition pair.
 * @param {string[]} site.dataSource       Contributing dataset per observation.
 * @param {number[]} site.validCount       Contributing pixels per observation.
 * @param {object}   site.values           { [variable]: number[] } aggregated values.
 * @param {object}   site.errors           { [variable]: number[] } aggregated errors.
 * @param {object} [options]
 * @param {string[]} [options.variable]    Variables to process, e.g. ['speed'].
 * @param {string[]} [options.sources]     Restrict to these datasets; empty means all.
 * @param {number}   [options.gapFill]     Max days of gap to bridge (server: gap_fill).
 * @param {number}   [options.winRaw]      Savitzky-Golay window for the raw points.
 * @param {number}   [options.winDaily]    Savitzky-Golay window for the daily average.
 * @param {number}   [options.poly]        Savitzky-Golay polynomial order.
 * @returns {object} { status, message, data }
 */
export function processSite(site, options = {}) {
  const {
    variable = ['speed'],
    sources = null,
    gapFill = 24,
    winRaw = 25,
    winDaily = 25,
    poly = 2,
  } = options;

  // 1. Restrict to the requested contributing datasets. data_source varies
  //    along time only, so this is a straight mask over the time axis.
  let keep = [];
  const wantedSources = sources && sources.length > 0 ? new Set(sources) : null;

  for (let i = 0; i < site.times.length; i++) {
    if (wantedSources && !wantedSources.has(String(site.dataSource[i]))) continue;
    keep.push(i);
  }

  // 2. Sort by time and drop duplicate timestamps, keeping the first, which is
  //    what df.sort_index() followed by groupby(level=0).first() does.
  keep.sort((a, b) => site.times[a] - site.times[b]);

  const seenTimes = new Set();
  keep = keep.filter((i) => {
    if (seenTimes.has(site.times[i])) return false;
    seenTimes.add(site.times[i]);
    return true;
  });

  const primary = variable[0];
  const hasAnyData = keep.some((i) => Number.isFinite(site.values[primary]?.[i]));
  if (keep.length === 0 || !hasAnyData) {
    return emptySiteResponse('error', 'No valid data or all selected sources masked/NaN', variable);
  }

  // 3. Gather the surviving observations into dense arrays.
  const times = keep.map((i) => site.times[i]);
  const dataSource = keep.map((i) => site.dataSource[i]);
  const validCount = keep.map((i) => site.validCount[i]);

  // Non-positive and missing separations are both replaced by 0.5 days. The
  // server's fillna(12.0) afterwards is unreachable, because the comparison
  // has already caught NaN, so we do not reproduce it.
  const timeSeparation = keep.map((i) => {
    const dt = site.timeSeparation[i];
    return Number.isFinite(dt) && dt > 0 ? dt : 0.5;
  });

  // 4. Build the timeline and an index for scattering onto it.
  const fullIndex = buildFullIndex(times);
  const positionOf = new Map(fullIndex.map((t, pos) => [t, pos]));

  const dtOnTimeline = reindexOntoTimeline(timeSeparation, times, positionOf, fullIndex.length);
  const sourceOnTimeline = reindexOntoTimeline(dataSource, times, positionOf, fullIndex.length, null);
  const countOnTimeline = reindexOntoTimeline(validCount, times, positionOf, fullIndex.length, 0);

  const outputData = {
    dates: fullIndex.map(formatTimestamp),
    dt: roundSeries(dtOnTimeline, 1),
    data_source: cleanNaNs(sourceOnTimeline),
    // count carries no gaps: absent days simply had no contributing pixels.
    count: countOnTimeline.map((v) => (Number.isFinite(v) ? Math.trunc(v) : 0)),
  };

  // 5. Process each requested variable independently.
  for (const name of variable) {
    const errors = keep.map((i) => site.errors[name]?.[i] ?? NaN);
    outputData[`${name}_error`] = roundSeries(
      reindexOntoTimeline(errors, times, positionOf, fullIndex.length), 2,
    );

    const observed = keep.map((i) => site.values[name]?.[i] ?? NaN);
    outputData[name] = processVariable(observed, {
      name, times, timeSeparation, fullIndex, positionOf, gapFill, winRaw, winDaily, poly,
    });
  }

  return { status: 'success', message: 'Data processed successfully.', data: outputData };
}

/**
 * Runs the outlier rejection, raw smoothing and weighted daily average for a
 * single variable, returning its { raw, smoothed } pair.
 */
function processVariable(observed, ctx) {
  const { name, times, timeSeparation, fullIndex, positionOf, gapFill, winRaw, winDaily, poly } = ctx;

  // 1. Reject values that are physically impossible outright.
  const values = Float64Array.from(observed, (v) => (Number.isFinite(v) ? v : NaN));

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;

    const impossible = name === 'speed'
      ? (v < SPEED_MIN || v > ABSOLUTE_MAX)
      : Math.abs(v) > ABSOLUTE_MAX;
    if (impossible) values[i] = NaN;
  }

  // 2. Reject local outliers: anything more than three rolling standard
  //    deviations from a five-sample rolling mean, with the series-wide
  //    standard deviation substituted where the window is too sparse to give
  //    one of its own.
  //
  //    Worth knowing: this test cannot actually fire. For a sample of n values
  //    the largest z-score any one of them can reach is (n-1)/sqrt(n), which
  //    for the five-sample window used here is 1.789 - the outlier inflates
  //    the very standard deviation it is being compared against. So the
  //    physical bounds above are what actually removes bad data.
  //
  //    It is reproduced faithfully rather than corrected, because changing it
  //    would alter every timeseries the application has ever published. If it
  //    is ever meant to do something, that is a change to make deliberately on
  //    the server first, with the window and threshold chosen together.
  const { mean: rollingMean, std: rollingStd } = rollingMeanStd(values, 5);

  let fallbackStd = nanStd(values);
  if (!Number.isFinite(fallbackStd) || fallbackStd === 0) fallbackStd = 1.0;

  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) continue;

    let spread = rollingStd[i];
    if (!Number.isFinite(spread) || spread === 0) spread = fallbackStd;

    if (Math.abs(values[i] - rollingMean[i]) > 3 * spread) values[i] = NaN;
  }

  // 3. Scatter onto the full timeline. Only these positions may carry a raw
  //    point in the output; the daily grid entries between them stay empty.
  const onTimeline = Float64Array.from(
    reindexOntoTimeline(values, times, positionOf, fullIndex.length),
  );
  const isObservation = Array.from(onTimeline, (v) => Number.isFinite(v));

  // --- STEP 1: RAW SMOOTHING (POINTS) ---
  // Bridge short gaps, fill the rest so the filter has a continuous signal,
  // smooth, then throw away everything that was not a real observation.
  const bridged = interpolateTime(onTimeline, fullIndex, { limit: gapFill });
  const continuous = interpolateTime(bridged, fullIndex, { limitDirection: 'both' });

  let rawSeries = onTimeline;
  const rawWindow = effectiveWindow(winRaw, fullIndex.length);

  if (rawWindow !== null && continuous.some(Number.isFinite)) {
    const smoothed = savgolFilter(continuous, rawWindow, poly);
    rawSeries = smoothed.map((v, i) => (isObservation[i] ? v : NaN));
  }

  // --- STEP 2: DURATION-WEIGHTED DAILY AVERAGE ---
  const dailyAverage = buildWeightedDailyAverage({
    values, rawSeries, times, timeSeparation, fullIndex, positionOf, gapFill,
  });

  // --- STEP 3: DAILY SMOOTHING AND GAP RE-MASKING ---
  const dailyBridged = interpolateTime(dailyAverage, fullIndex, { limit: gapFill });
  const dailyContinuous = interpolateTime(dailyBridged, fullIndex, { limitDirection: 'both' });

  let smoothedSeries = dailyAverage;
  const dailyWindow = effectiveWindow(winDaily, fullIndex.length);

  if (dailyWindow !== null && dailyContinuous.some(Number.isFinite)) {
    const smoothed = savgolFilter(dailyContinuous, dailyWindow, poly);
    // Re-open the gaps the smoother has just papered over: a filter cannot
    // invent data across a gap wider than gapFill.
    smoothedSeries = smoothed.map((v, i) => (Number.isFinite(dailyBridged[i]) ? v : NaN));
  }

  return {
    raw: roundSeries(rawSeries, 2),
    smoothed: roundSeries(smoothedSeries, 2),
  };
}

/**
 * Builds the daily average, weighting each observation by 1/separation so that
 * a tightly-spaced acquisition pair counts for more than a widely-spaced one.
 *
 * Each observation is smeared across the days its acquisition pair spans
 * (capped at gapFill so a year-long pair does not flatten the whole series),
 * and overlapping smears are combined as a weighted mean.
 */
function buildWeightedDailyAverage(ctx) {
  const { values, rawSeries, times, timeSeparation, fullIndex, positionOf, gapFill } = ctx;

  const weightedSum = new Map();
  const weightSum = new Map();

  for (let i = 0; i < times.length; i++) {
    if (!Number.isFinite(values[i])) continue;

    // 1. Prefer the smoothed value at this observation, falling back to the
    //    measured one wherever smoothing produced nothing.
    const pos = positionOf.get(times[i]);
    const smoothedHere = pos === undefined ? NaN : rawSeries[pos];
    const value = Number.isFinite(smoothedHere) ? smoothedHere : values[i];

    // 2. Sub-daily pairs would otherwise get enormous weight, so the server
    //    floors the separation at one day before inverting it.
    const separation = timeSeparation[i];
    const weight = 1.0 / (Number.isFinite(separation) && separation >= 1 ? separation : 1.0);

    // 3. Smear across the days the acquisition pair covers.
    const capped = Math.min(separation, gapFill);
    const halfSpan = (capped / 2) * MS_PER_DAY;
    const firstDay = floorDay(times[i] - halfSpan);
    const lastDay = ceilDay(times[i] + halfSpan);

    for (let day = firstDay; day <= lastDay; day += MS_PER_DAY) {
      weightedSum.set(day, (weightedSum.get(day) || 0) + value * weight);
      weightSum.set(day, (weightSum.get(day) || 0) + weight);
    }
  }

  // 4. Collapse to a weighted mean per day and scatter onto the timeline.
  //    Timeline entries that are not whole days hold no daily average.
  const out = new Float64Array(fullIndex.length);
  out.fill(NaN);

  for (const [day, total] of weightedSum) {
    const pos = positionOf.get(day);
    if (pos === undefined) continue;

    const totalWeight = weightSum.get(day);
    if (totalWeight) out[pos] = total / totalWeight;
  }

  return out;
}
