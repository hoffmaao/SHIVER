/**
 * parity.integration.test.js
 *
 * End-to-end check that a client-side extraction agrees with the production
 * backend for the same query, reading live data from Source Cooperative.
 *
 * This is the test that would catch a change to the Zarr stores - a re-chunk,
 * a new epoch, a renamed variable - which the recorded fixture in
 * timeseriesProcessing.test.js cannot see. It needs both the CDN and the
 * production VM, so it is opt-in rather than part of the default run:
 *
 *     SHIVER_LIVE_TESTS=1 npm test
 *
 * Expect it to take a couple of minutes and to pull a few hundred megabytes:
 * every fresh 64x64 tile is roughly 15 MB of velocities plus 9 MB of errors.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { extractTimeseries } from '../src/utils/shiverTimeseries.js';

const API = 'https://shiver.shef.ac.uk';
const LIVE = process.env.SHIVER_LIVE_TESTS === '1';

// Timeouts are generous: a cold chunk is a large download.
const TIMEOUT = 240000;

const CASES = [
  {
    name: 'Jakobshavn Isbrae, single pixel',
    roi: [[69.15, -49.55]],
    settings: { buffer: 0 },
  },
  {
    name: 'Jakobshavn Isbrae, 500 m buffer',
    roi: [[69.15, -49.55]],
    settings: { buffer: 500 },
  },
  {
    name: 'Helheim Glacier, filtered to two sources',
    roi: [[66.4, -38.2]],
    settings: { buffer: 500, sources: ['SHIFT', 'ESA_CCI_Sentinel-1'] },
  },
  {
    name: 'Greenland interior, slow flow',
    roi: [[72.5, -40.0]],
    settings: { buffer: 500 },
  },
];

const BASE_SETTINGS = {
  buffer: 500,
  variable: ['speed'],
  sources: [],
  gap_fill: 24,
  win_raw: 25,
  win_daily: 25,
  poly: 2,
};

/** Asks the production backend for the same query. */
async function askServer(roi, settings) {
  const response = await fetch(`${API}/api/timeseries/multi/json`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ roi, ...settings }),
  });
  if (!response.ok) throw new Error(`API returned ${response.status}`);

  return (await response.json()).Site_0;
}

/**
 * Compares a series against the server's, allowing one unit in the last
 * rounded decimal place but requiring the gap pattern to line up exactly.
 * Returns how many values disagreed at all.
 */
function compareSeries(actual, expected, label, tolerance = 0.01) {
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

    const difference = Math.abs(a - e);
    expect(
      difference,
      `${label}[${i}]: server ${e}, we produced ${a}`,
    ).toBeLessThanOrEqual(tolerance + 1e-9);

    if (difference > 1e-9) differing++;
  }
  return differing;
}

describe.skipIf(!LIVE)('client extraction against the live production API', () => {
  beforeAll(() => {
    console.log('Running live parity tests - this downloads several hundred MB.');
  });

  for (const testCase of CASES) {
    it(`matches the server for ${testCase.name}`, async () => {
      const settings = { ...BASE_SETTINGS, ...testCase.settings };

      const [server, clientResults] = await Promise.all([
        askServer(testCase.roi, settings),
        extractTimeseries(testCase.roi, settings),
      ]);
      const client = clientResults.Site_0;

      // A site the server could not process should fail on our side too, for
      // the same reason.
      if (server.status !== 'success') {
        expect(client.status).toBe(server.status);
        return;
      }

      expect(client.status).toBe('success');
      expect(client.data.dates).toEqual(server.data.dates);

      // Metadata involves no smoothing, so it must agree exactly.
      compareSeries(client.data.dt, server.data.dt, 'dt', 0);
      compareSeries(client.data.count, server.data.count, 'count', 0);
      compareSeries(client.data.data_source, server.data.data_source, 'data_source', 0);
      compareSeries(client.data.speed_error, server.data.speed_error, 'speed_error', 0);

      const rawDiffs = compareSeries(client.data.speed.raw, server.data.speed.raw, 'speed.raw');
      const smoothDiffs = compareSeries(
        client.data.speed.smoothed, server.data.speed.smoothed, 'speed.smoothed',
      );

      // Last-place disagreements should stay a small minority.
      const total = server.data.dates.length;
      expect(rawDiffs / total).toBeLessThan(0.05);
      expect(smoothDiffs / total).toBeLessThan(0.05);
    }, TIMEOUT);
  }

  it('reports a location outside coverage the same way the server does', async () => {
    // Mid-Atlantic: on the Greenland side of the equator, far off the grid.
    const roi = [[45.0, -30.0]];
    const [server, clientResults] = await Promise.all([
      askServer(roi, BASE_SETTINGS),
      extractTimeseries(roi, BASE_SETTINGS),
    ]);

    expect(clientResults.Site_0.status).toBe('error');
    expect(server.status).toBe('error');
    expect(clientResults.Site_0.data.dates).toEqual([]);
  }, TIMEOUT);

  it('attaches the same meta block the server does', async () => {
    const settings = { ...BASE_SETTINGS, buffer: 0 };
    const roi = [[69.15, -49.55]];

    const [server, clientResults] = await Promise.all([
      askServer(roi, settings),
      extractTimeseries(roi, settings),
    ]);

    // MultiSourceMap.vue reads these for its labels, exports and CSV headers.
    expect(clientResults.Site_0.meta).toEqual(server.meta);
  }, TIMEOUT);
});
