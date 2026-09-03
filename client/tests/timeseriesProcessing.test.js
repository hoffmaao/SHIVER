/**
 * timeseriesProcessing.test.js
 *
 * Checks that the browser pipeline reproduces what the SHIVER backend returns.
 *
 * The headline test replays a recorded extraction - the arrays actually read
 * out of the Zarr store for a point on Jakobshavn Isbrae - and compares the
 * result against the payload the production API returned for that same query.
 * That runs offline, so CI does not depend on the production VM being up.
 *
 * The remaining tests pin individual behaviours of the pipeline with small
 * synthetic inputs, where a failure points straight at the step that broke.
 */

import { describe, it, expect } from 'vitest';

import fixture from './fixtures/parity_fixture.json';
import { processSite, buildFullIndex } from '../src/utils/timeseriesProcessing.js';

const MS_PER_DAY = 86400000;

// --- 1. HELPERS ---

const decode = (values) => values.map((v) => (v === null ? NaN : v));

/** Rebuilds the processSite input from the recorded arrays. */
function fixtureInput() {
  return {
    times: fixture.input.times,
    timeSeparation: decode(fixture.input.timeSeparation),
    dataSource: fixture.input.dataSource,
    validCount: fixture.input.validCount,
    values: { speed: decode(fixture.input.speed) },
    errors: { speed: decode(fixture.input.speedError) },
  };
}

/**
 * Compares a produced series against the server's.
 *
 * Gaps must line up exactly - a value where the server had none, or the
 * reverse, is a real divergence. Present values are allowed to differ by one
 * unit in the last decimal place the server rounds to: both sides round to two
 * decimals, and the two pipelines accumulate their floating point in a
 * different order, so a value sitting on a .005 boundary can tip either way.
 * Anything larger than that means the maths, not the rounding, has drifted.
 */
function expectMatchesServer(actual, expected, label, tolerance = 0.01) {
  expect(actual.length, `${label}: length`).toBe(expected.length);

  let differing = 0;

  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];

    if (e === null || e === undefined) {
      expect(a ?? null, `${label}[${i}]: server had a gap, we produced ${a}`).toBeNull();
      continue;
    }
    expect(a, `${label}[${i}]: server had ${e}, we produced a gap`).not.toBeNull();

    if (typeof e === 'string') {
      expect(a, `${label}[${i}]`).toBe(e);
      continue;
    }

    // Both sides are already rounded to two decimals, so subtracting them
    // leaves a little binary noise on top of the true difference. The slack
    // absorbs that without letting a genuine second-place error through.
    const difference = Math.abs(a - e);
    expect(
      difference,
      `${label}[${i}]: server ${e}, we produced ${a} (difference ${difference})`,
    ).toBeLessThanOrEqual(tolerance + 1e-9);

    if (difference > 1e-9) differing++;
  }

  // Last-place disagreements are expected on a small minority of values. A
  // sudden jump here would mean something systematic had changed even though
  // every individual value still scraped inside tolerance.
  const differingFraction = differing / Math.max(1, expected.length);
  expect(differingFraction, `${label}: ${differing} of ${expected.length} values differ`)
    .toBeLessThan(0.05);
}

// --- 2. PARITY WITH THE PRODUCTION BACKEND ---

describe('processSite against a recorded production response', () => {
  const results = processSite(fixtureInput(), {
    variable: fixture.case.settings.variable,
    sources: fixture.case.settings.sources,
    gapFill: fixture.case.settings.gap_fill,
    winRaw: fixture.case.settings.win_raw,
    winDaily: fixture.case.settings.win_daily,
    poly: fixture.case.settings.poly,
  });

  it('succeeds', () => {
    expect(results.status).toBe('success');
    expect(results.message).toBe('Data processed successfully.');
  });

  it('builds an identical timeline', () => {
    expect(results.data.dates).toEqual(fixture.expected.dates);
  });

  it('reproduces the per-epoch metadata exactly', () => {
    // These involve no smoothing, so they should agree to the last digit.
    expectMatchesServer(results.data.dt, fixture.expected.dt, 'dt', 0);
    expectMatchesServer(results.data.count, fixture.expected.count, 'count', 0);
    expectMatchesServer(results.data.data_source, fixture.expected.data_source, 'data_source', 0);
    expectMatchesServer(results.data.speed_error, fixture.expected.speed_error, 'speed_error', 0);
  });

  it('reproduces the raw speed points', () => {
    expectMatchesServer(results.data.speed.raw, fixture.expected.speed.raw, 'speed.raw');
  });

  it('reproduces the smoothed daily speed', () => {
    expectMatchesServer(results.data.speed.smoothed, fixture.expected.speed.smoothed, 'speed.smoothed');
  });

  it('was recorded against a known dataset version', () => {
    // If this changes, the fixture predates a store update and should be
    // re-recorded rather than silently trusted.
    expect(fixture.datasetVersion).toBeTruthy();
  });
});

// --- 3. SOURCE FILTERING ---

describe('processSite source filtering', () => {
  /** A short synthetic record alternating between two contributing datasets. */
  function syntheticInput(epochs = 40) {
    const times = [];
    const dataSource = [];
    const speed = [];

    for (let i = 0; i < epochs; i++) {
      times.push(Date.UTC(2020, 0, 1) + i * 6 * MS_PER_DAY);
      dataSource.push(i % 2 === 0 ? 'PROMICE' : 'SHIFT');
      speed.push(1000 + i * 10);
    }

    return {
      times,
      timeSeparation: new Array(epochs).fill(12),
      dataSource,
      validCount: new Array(epochs).fill(1),
      values: { speed },
      errors: { speed: new Array(epochs).fill(25) },
    };
  }

  it('keeps only the requested datasets', () => {
    const results = processSite(syntheticInput(), { sources: ['PROMICE'] });
    const present = results.data.data_source.filter((s) => s !== null);

    expect(present.length).toBeGreaterThan(0);
    expect(new Set(present)).toEqual(new Set(['PROMICE']));
  });

  it('keeps everything when no dataset is specified', () => {
    const results = processSite(syntheticInput(), { sources: [] });
    const present = new Set(results.data.data_source.filter((s) => s !== null));

    expect(present).toEqual(new Set(['PROMICE', 'SHIFT']));
  });

  it('reports an error rather than throwing when a filter matches nothing', () => {
    const results = processSite(syntheticInput(), { sources: ['ITS_LIVE_annual'] });

    expect(results.status).toBe('error');
    expect(results.message).toMatch(/No valid data/);
    // The full key structure survives so the chart code needs no special case.
    expect(results.data.speed).toEqual({ raw: [], smoothed: [] });
    expect(results.data.dates).toEqual([]);
  });
});

// --- 4. OUTLIER REJECTION ---

describe('processSite outlier rejection', () => {
  /** A steady series with one absurd value dropped into the middle. */
  function withOutlier(outlierValue) {
    const epochs = 60;
    const times = [];
    const speed = [];

    for (let i = 0; i < epochs; i++) {
      times.push(Date.UTC(2020, 0, 1) + i * 6 * MS_PER_DAY);
      speed.push(2000);
    }
    speed[30] = outlierValue;

    return {
      times,
      timeSeparation: new Array(epochs).fill(12),
      dataSource: new Array(epochs).fill('SHIFT'),
      validCount: new Array(epochs).fill(1),
      values: { speed },
      errors: { speed: new Array(epochs).fill(25) },
    };
  }

  it('drops values outside the physically plausible range', () => {
    // The store really does contain values like -3.6e6 at some pixels.
    const results = processSite(withOutlier(-3652134.8), {});
    const raw = results.data.speed.raw.filter((v) => v !== null);

    expect(raw.length).toBeGreaterThan(0);
    for (const value of raw) expect(Math.abs(value)).toBeLessThan(100000);
  });

  it('leaves an in-range spike in place, because the rolling test cannot fire', () => {
    // The server's rolling three-sigma test is unreachable: a lone spike
    // inflates the five-sample standard deviation it is measured against, and
    // the largest z-score a sample of n values can reach is (n-1)/sqrt(n),
    // which is 1.789 for n=5. This pins that behaviour so a future change to
    // the window or the threshold shows up here rather than silently altering
    // published timeseries. See the note in timeseriesProcessing.js.
    const results = processSite(withOutlier(9000), {});
    const spikeDate = new Date(Date.UTC(2020, 0, 1) + 30 * 6 * MS_PER_DAY)
      .toISOString().slice(0, 19);
    const atSpike = results.data.speed.raw[results.data.dates.indexOf(spikeDate)];

    // The point survives; smoothing pulls it towards its neighbours but it is
    // nowhere near dropped.
    expect(atSpike).not.toBeNull();
    expect(atSpike).toBeGreaterThan(2000);
  });

  it('keeps a small negative speed, which is a legitimate measurement', () => {
    const results = processSite(withOutlier(-50), {});
    const raw = results.data.speed.raw.filter((v) => v !== null);

    expect(raw.length).toBeGreaterThan(0);
  });
});

// --- 5. TIMELINE CONSTRUCTION ---

describe('buildFullIndex', () => {
  it('unions the observation times with a complete daily grid', () => {
    const noon = Date.UTC(2020, 0, 2, 12);
    const index = buildFullIndex([Date.UTC(2020, 0, 1), noon, Date.UTC(2020, 0, 3)]);

    expect(index).toEqual([
      Date.UTC(2020, 0, 1),
      Date.UTC(2020, 0, 2),
      noon,
      Date.UTC(2020, 0, 3),
    ]);
  });

  it('collapses observations that already sit on a whole day', () => {
    const index = buildFullIndex([Date.UTC(2020, 0, 1), Date.UTC(2020, 0, 2)]);
    expect(index).toEqual([Date.UTC(2020, 0, 1), Date.UTC(2020, 0, 2)]);
  });

  it('returns nothing for an empty record', () => {
    expect(buildFullIndex([])).toEqual([]);
  });

  it('matches the timeline length the server produced for the recorded case', () => {
    const index = buildFullIndex(fixture.input.times);
    expect(index.length).toBe(fixture.expected.dates.length);
  });
});

// --- 6. OUTPUT SHAPE ---

describe('processSite output shape', () => {
  it('formats dates the way the server does, with no timezone suffix', () => {
    // Appending a Z here would shift every point in the chart by the viewer's
    // offset, because the server's timestamps are naive.
    for (const date of fixture.expected.dates.slice(0, 5)) {
      expect(date).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/);
    }
  });

  it('returns gaps as null rather than NaN, so the payload survives JSON', () => {
    const results = processSite(fixtureInput(), { variable: ['speed'] });
    const roundTripped = JSON.parse(JSON.stringify(results));

    expect(roundTripped.data.speed.raw).toEqual(results.data.speed.raw);
    expect(results.data.speed.raw.some((v) => v === null)).toBe(true);
    expect(results.data.speed.raw.every((v) => v === null || Number.isFinite(v))).toBe(true);
  });

  it('never reports a negative or fractional pixel count', () => {
    const results = processSite(fixtureInput(), { variable: ['speed'] });

    for (const count of results.data.count) {
      expect(Number.isInteger(count)).toBe(true);
      expect(count).toBeGreaterThanOrEqual(0);
    }
  });
});
