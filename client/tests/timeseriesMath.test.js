/**
 * timeseriesMath.test.js
 *
 * Asserts that the JavaScript numerical primitives reproduce the pandas/scipy/
 * numpy behaviours that server/utils/extract_multi_zarr_ts.py relies on.
 *
 * Every expectation in here is ground truth captured from the real libraries by
 * client/tests/fixtures/generate_math_fixtures.py, not a hand-written guess.
 */

import { describe, it, expect } from 'vitest';
import fixtures from './fixtures/math_fixtures.json';

import {
  npRound,
  interpolateTime,
  rollingMeanStd,
  nanStd,
  savgolFilter,
  savgolCoeffs,
  effectiveWindow,
} from '../src/utils/timeseriesMath.js';

// --- 1. HELPERS ---

/** JSON cannot carry NaN, so the fixtures use null in its place. */
const decode = (values) => values.map((v) => (v === null ? NaN : v));

/**
 * Compares two series that may contain NaN. Vitest's toBeCloseTo does not
 * treat NaN as equal to NaN, and a mismatched gap pattern is exactly the kind
 * of bug these tests exist to catch, so gaps are compared explicitly.
 */
function expectSeriesClose(actual, expected, tolerance, label) {
  expect(actual.length, `${label}: length`).toBe(expected.length);

  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];

    if (!Number.isFinite(e)) {
      expect(Number.isFinite(a), `${label}[${i}]: expected a gap, got ${a}`).toBe(false);
      continue;
    }
    expect(Number.isFinite(a), `${label}[${i}]: expected ${e}, got a gap`).toBe(true);

    // Scale the tolerance with the magnitude so that speeds in the tens of
    // thousands are held to the same relative accuracy as values near zero.
    const allowed = tolerance * Math.max(1, Math.abs(e));
    expect(Math.abs(a - e), `${label}[${i}]: expected ${e}, got ${a}`).toBeLessThanOrEqual(allowed);
  }
}

// --- 2. ROUNDING ---

describe('npRound', () => {
  it('matches numpy.round across the generated value sweep', () => {
    for (const testCase of fixtures.rounding) {
      const values = decode(testCase.values);
      const expected = decode(testCase.expected);

      for (let i = 0; i < values.length; i++) {
        expect(
          npRound(values[i], testCase.decimals),
          `round(${values[i]}, ${testCase.decimals})`,
        ).toBe(expected[i]);
      }
    }
  });

  it('rounds exact halves to even rather than away from zero', () => {
    // Math.round would give 0.13 and 0.26 here; numpy gives 0.12 and 0.26.
    expect(npRound(0.125, 2)).toBe(0.12);
    expect(npRound(0.255, 2)).toBe(0.26);
    expect(npRound(2.5, 0)).toBe(2);
    expect(npRound(3.5, 0)).toBe(4);
  });

  it('propagates gaps instead of turning them into zero', () => {
    expect(npRound(NaN, 2)).toBeNaN();
    expect(npRound(Infinity, 2)).toBeNaN();
  });
});

// --- 3. INTERPOLATION ---

describe('interpolateTime', () => {
  for (const testCase of fixtures.interpolateTime) {
    it(`matches pandas interpolate(method='time') for ${testCase.name}`, () => {
      const options = { limitDirection: testCase.options.limitDirection };
      if (testCase.options.limit !== null && testCase.options.limit !== undefined) {
        options.limit = testCase.options.limit;
      }

      const actual = interpolateTime(decode(testCase.values), testCase.times, options);
      expectSeriesClose(actual, decode(testCase.expected), 1e-9, testCase.name);
    });
  }

  it('extrapolates flat past the final observation rather than continuing the trend', () => {
    // A linear extension would reach 30; pandas holds the last value at 20.
    const times = [0, 1, 2, 3].map((d) => d * 86400000);
    const result = interpolateTime([10, 20, NaN, NaN], times, { limit: 24 });

    expect(result[2]).toBe(20);
    expect(result[3]).toBe(20);
  });

  it('leaves a leading gap empty in the default forward direction', () => {
    const times = [0, 1, 2, 3].map((d) => d * 86400000);
    const result = interpolateTime([NaN, NaN, 5, 7], times, { limit: 24 });

    expect(result[0]).toBeNaN();
    expect(result[1]).toBeNaN();
    expect(result[2]).toBe(5);
  });

  it('does not modify the array it is given', () => {
    const input = [1, NaN, 3];
    const times = [0, 86400000, 172800000];
    interpolateTime(input, times, { limit: 24 });

    expect(input[1]).toBeNaN();
  });
});

// --- 4. ROLLING STATISTICS ---

describe('rollingMeanStd', () => {
  for (const testCase of fixtures.rolling) {
    it(`matches pandas rolling(center=True, min_periods=1) for ${testCase.name}`, () => {
      const { mean, std } = rollingMeanStd(decode(testCase.values), testCase.window);

      expectSeriesClose(mean, decode(testCase.expectedMean), 1e-9, `${testCase.name} mean`);
      expectSeriesClose(std, decode(testCase.expectedStd), 1e-9, `${testCase.name} std`);
    });

    it(`matches pandas Series.std() for ${testCase.name}`, () => {
      const expected = testCase.expectedOverallStd;
      const actual = nanStd(decode(testCase.values));

      if (expected === null) expect(actual).toBeNaN();
      else expect(actual).toBeCloseTo(expected, 9);
    });
  }

  it('returns a gap where a window holds a single observation, not zero', () => {
    // ddof=1 makes the sample standard deviation of one value undefined. If
    // this returned 0 the outlier test downstream would reject every point.
    const { std } = rollingMeanStd([NaN, NaN, 5, NaN, NaN], 5);
    expect(std[2]).toBeNaN();
  });
});

// --- 5. SAVITZKY-GOLAY ---

describe('savgolFilter', () => {
  for (const testCase of fixtures.savgol) {
    it(`matches scipy savgol_filter for ${testCase.name}`, () => {
      const actual = savgolFilter(decode(testCase.values), testCase.window, testCase.poly);
      expectSeriesClose(actual, decode(testCase.expected), 1e-9, testCase.name);
    });

    it(`matches scipy savgol_coeffs for ${testCase.name}`, () => {
      const actual = savgolCoeffs(testCase.window, testCase.poly);
      expectSeriesClose(actual, decode(testCase.expectedCoeffs), 1e-9, `${testCase.name} coeffs`);
    });
  }

  it('reproduces the edge samples, not just the interior', () => {
    // The first and last (window-1)/2 samples come from a polynomial fit over
    // the end windows. Padding the signal instead would drift here.
    const noisy = fixtures.savgol.find((c) => c.name === 'noisy_w25_p2');
    const actual = savgolFilter(decode(noisy.values), noisy.window, noisy.poly);
    const expected = decode(noisy.expected);
    const half = (noisy.window - 1) / 2;

    for (let i = 0; i < half; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 6);
      expect(actual[actual.length - 1 - i]).toBeCloseTo(expected[expected.length - 1 - i], 6);
    }
  });

  it('rejects window sizes the server would never produce', () => {
    expect(() => savgolFilter([1, 2, 3], 5, 2)).toThrow(/exceeds input length/);
    expect(() => savgolFilter([1, 2, 3, 4], 4, 2)).toThrow(/must be odd/);
    expect(() => savgolFilter([1, 2, 3, 4, 5], 3, 3)).toThrow(/must be less than/);
  });
});

describe('effectiveWindow', () => {
  it('clamps to the series length and forces an odd window', () => {
    expect(effectiveWindow(25, 400)).toBe(25);   // comfortably long enough
    expect(effectiveWindow(25, 20)).toBe(19);    // shrunk to 20, then made odd
    expect(effectiveWindow(25, 25)).toBe(25);
    expect(effectiveWindow(24, 400)).toBe(23);   // even request made odd
  });

  it('reports that smoothing should be skipped for very short series', () => {
    expect(effectiveWindow(25, 2)).toBeNull();
    expect(effectiveWindow(25, 1)).toBeNull();
    expect(effectiveWindow(25, 3)).toBe(3);
  });
});
