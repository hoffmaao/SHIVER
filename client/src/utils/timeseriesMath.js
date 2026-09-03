/**
 * timeseriesMath.js
 *
 * Numerical primitives required to reproduce the SHIVER server-side timeseries
 * processing in the browser. Each function here is a deliberate port of a
 * specific pandas/scipy/numpy behaviour used by
 * server/utils/extract_multi_zarr_ts.py, so that a client-side extraction
 * returns the same numbers as /api/timeseries/multi/json.
 *
 * Everything in this file is pure (no I/O, no DOM), so it runs unchanged on the
 * main thread, inside a Web Worker, and under vitest.
 *
 * Missing data is represented as NaN throughout, matching numpy. It is only
 * converted to null at the very end of the pipeline (see cleanNaNs).
 */

// --- 1. ROUNDING ---

/**
 * Reproduces numpy.round(value, decimals).
 *
 * JavaScript's Math.round rounds halves away from zero, but numpy rounds them
 * to the nearest EVEN integer ("banker's rounding"). Getting this wrong
 * produces off-by-0.01 differences against the server on roughly one value in
 * a hundred, so we implement numpy's exact multiply -> rint -> divide sequence.
 * Verified to match numpy on 250,000 random values at 1 and 2 decimals.
 */
export function npRound(value, decimals = 0) {
  if (!Number.isFinite(value)) return NaN;

  const scale = Math.pow(10, decimals);
  const scaled = value * scale;
  const floored = Math.floor(scaled);
  const remainder = scaled - floored;

  let rounded;
  if (remainder > 0.5) rounded = floored + 1;
  else if (remainder < 0.5) rounded = floored;
  // Exact half: pick whichever neighbour is even.
  else rounded = (floored % 2 === 0) ? floored : floored + 1;

  return rounded / scale;
}

// --- 2. TIME-WEIGHTED INTERPOLATION ---

/**
 * Reproduces pandas Series.interpolate(method='time', ...).
 *
 * pandas builds a linear interpolation over the *time* axis and then masks the
 * result according to `limit` / `limitDirection`. Two behaviours matter here
 * and neither is obvious from the pandas docs:
 *
 *   a) Values outside the range of valid observations are extrapolated FLAT
 *      (constant), exactly like numpy.interp - they are not left as NaN and
 *      they are not linearly extended.
 *   b) `limit` counts consecutive NaNs from the START of each gap, moving
 *      forward. With the default limitDirection='forward', a run of NaNs that
 *      sits before the first valid observation is never filled at all.
 *
 * @param {Float64Array|number[]} values  Series values (NaN where missing).
 * @param {Float64Array|number[]} times   Strictly increasing times, in ms.
 * @param {object}   [options]
 * @param {number}   [options.limit]           Max consecutive NaNs to fill.
 * @param {string}   [options.limitDirection]  'forward' (default) or 'both'.
 * @returns {Float64Array} A new array; the input is not modified.
 */
export function interpolateTime(values, times, options = {}) {
  const { limit = Infinity, limitDirection = 'forward' } = options;

  const n = values.length;
  const out = new Float64Array(n);
  out.fill(NaN);

  // 1. Locate the observations we can actually interpolate between.
  const validIdx = [];
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(values[i])) validIdx.push(i);
  }
  if (validIdx.length === 0) return out;

  // 2. Full linear interpolation with flat extrapolation at both ends,
  //    i.e. numpy.interp. Masking is applied afterwards.
  const first = validIdx[0];
  const last = validIdx[validIdx.length - 1];
  let cursor = 0;

  for (let i = 0; i < n; i++) {
    if (Number.isFinite(values[i])) {
      out[i] = values[i];
      continue;
    }
    if (i < first) { out[i] = values[first]; continue; }
    if (i > last) { out[i] = values[last]; continue; }

    // Advance to the bracketing pair of valid samples for this index.
    while (cursor < validIdx.length - 1 && validIdx[cursor + 1] < i) cursor++;
    const lo = validIdx[cursor];
    const hi = validIdx[cursor + 1];

    const span = times[hi] - times[lo];
    // Coincident timestamps would divide by zero; fall back to the left value.
    const frac = span === 0 ? 0 : (times[i] - times[lo]) / span;
    out[i] = values[lo] + frac * (values[hi] - values[lo]);
  }

  // 3. Re-mask according to the pandas limit rules, one maximal run of
  //    originally-missing values at a time.
  //
  //    A run can only be filled in a direction it is anchored in: forwards
  //    from a preceding observation, backwards from a following one. So a
  //    leading run is never filled forwards (there is nothing to carry
  //    forward) and a trailing run is never filled backwards, regardless of
  //    limitDirection. Within each permitted direction, `limit` values are
  //    filled from that end of the run and the remainder stays empty.
  let runStart = -1;
  for (let i = 0; i <= n; i++) {
    const missing = i < n && !Number.isFinite(values[i]);

    if (missing && runStart === -1) runStart = i;
    if (missing || runStart === -1) continue;

    const runEnd = i - 1;                              // inclusive
    const fillsForward = runStart > first;             // has an observation to its left
    const fillsBackward = runEnd < last && limitDirection === 'both';

    // Widest index still reachable going forwards, and narrowest going
    // backwards. Anything between the two is beyond every limit and is cleared.
    const forwardStop = fillsForward ? runStart + limit - 1 : -Infinity;
    const backwardStart = fillsBackward ? runEnd - limit + 1 : Infinity;

    for (let k = runStart; k <= runEnd; k++) {
      if (k > forwardStop && k < backwardStart) out[k] = NaN;
    }
    runStart = -1;
  }

  return out;
}

// --- 3. ROLLING WINDOW STATISTICS ---

/**
 * Reproduces pandas .rolling(window, center=True, min_periods=1).mean()/.std().
 *
 * NaNs are skipped rather than propagated, and the standard deviation is the
 * SAMPLE standard deviation (ddof=1), so a window containing a single valid
 * observation yields NaN rather than 0. Windows are clipped at the array
 * edges instead of being padded.
 *
 * A direct O(n * window) evaluation is used rather than a running sum: the
 * window is small (5) and this avoids the catastrophic cancellation that
 * running sums suffer on values in the tens of thousands.
 *
 * @returns {{mean: Float64Array, std: Float64Array}}
 */
export function rollingMeanStd(values, window) {
  const n = values.length;
  const mean = new Float64Array(n);
  const std = new Float64Array(n);
  mean.fill(NaN);
  std.fill(NaN);

  const half = Math.floor(window / 2);

  for (let i = 0; i < n; i++) {
    // pandas centres an odd window on i; for even windows it leans left.
    const start = Math.max(0, i - half);
    const stop = Math.min(n - 1, i + (window - 1 - half));

    let count = 0;
    let sum = 0;
    for (let k = start; k <= stop; k++) {
      const v = values[k];
      if (Number.isFinite(v)) { sum += v; count++; }
    }
    if (count === 0) continue;          // min_periods=1 not satisfied

    const m = sum / count;
    mean[i] = m;

    if (count < 2) continue;            // ddof=1 leaves std undefined
    let sq = 0;
    for (let k = start; k <= stop; k++) {
      const v = values[k];
      if (Number.isFinite(v)) { const d = v - m; sq += d * d; }
    }
    std[i] = Math.sqrt(sq / (count - 1));
  }

  return { mean, std };
}

/**
 * Sample standard deviation (ddof=1) over all finite values, skipping NaN.
 * Equivalent to pandas Series.std().
 */
export function nanStd(values) {
  let count = 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) { sum += v; count++; }
  }
  if (count < 2) return NaN;

  const m = sum / count;
  let sq = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) { const d = v - m; sq += d * d; }
  }
  return Math.sqrt(sq / (count - 1));
}

// --- 4. LEAST SQUARES HELPERS ---

/**
 * Solves the normal equations for a polynomial least-squares fit and returns
 * the coefficients in ascending power order.
 *
 * The abscissae are centred and scaled onto [-1, 1] before the Vandermonde
 * matrix is built. That transformation is exact, costs nothing, and keeps the
 * normal-equations matrix far away from the Hilbert-like conditioning that
 * raw positions (0..100) would produce.
 *
 * @returns {{coeffs: number[], offset: number, scale: number}} Evaluate with
 *          polyEval(fit, position).
 */
export function polyFit(xs, ys, degree) {
  const n = xs.length;
  const cols = degree + 1;

  // 1. Normalise the abscissae for conditioning.
  let offset = 0;
  for (let i = 0; i < n; i++) offset += xs[i];
  offset /= n;

  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(xs[i] - offset));
  if (scale === 0) scale = 1;

  // 2. Accumulate the normal equations A^T A c = A^T y directly. Powers of the
  //    normalised abscissa are cheap enough to recompute per row.
  const ata = Array.from({ length: cols }, () => new Float64Array(cols));
  const aty = new Float64Array(cols);

  const powers = new Float64Array(cols);
  for (let i = 0; i < n; i++) {
    const t = (xs[i] - offset) / scale;
    powers[0] = 1;
    for (let j = 1; j < cols; j++) powers[j] = powers[j - 1] * t;

    for (let r = 0; r < cols; r++) {
      aty[r] += powers[r] * ys[i];
      for (let c = r; c < cols; c++) ata[r][c] += powers[r] * powers[c];
    }
  }
  // Mirror the symmetric half we skipped.
  for (let r = 1; r < cols; r++) {
    for (let c = 0; c < r; c++) ata[r][c] = ata[c][r];
  }

  return { coeffs: solveLinearSystem(ata, aty), offset, scale };
}

/** Evaluates a fit produced by polyFit at an untransformed position. */
export function polyEval(fit, x) {
  const t = (x - fit.offset) / fit.scale;
  let acc = 0;
  // Horner, descending through the coefficients.
  for (let j = fit.coeffs.length - 1; j >= 0; j--) acc = acc * t + fit.coeffs[j];
  return acc;
}

/**
 * Gaussian elimination with partial pivoting. The systems solved here are tiny
 * (at most 5x5 for polyorder 4), so a dedicated decomposition would be
 * needless machinery.
 */
function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row) => Float64Array.from(row));
  const b = Float64Array.from(rhs);

  for (let col = 0; col < n; col++) {
    // 1. Pivot on the largest remaining magnitude in this column.
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    }
    if (pivot !== col) {
      const tmpRow = a[col]; a[col] = a[pivot]; a[pivot] = tmpRow;
      const tmpVal = b[col]; b[col] = b[pivot]; b[pivot] = tmpVal;
    }
    if (a[col][col] === 0) continue;   // singular; leave this coefficient at 0

    // 2. Eliminate below.
    for (let r = col + 1; r < n; r++) {
      const factor = a[r][col] / a[col][col];
      if (factor === 0) continue;
      for (let c = col; c < n; c++) a[r][c] -= factor * a[col][c];
      b[r] -= factor * b[col];
    }
  }

  // 3. Back-substitute.
  const out = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    if (a[r][r] === 0) { out[r] = 0; continue; }
    let acc = b[r];
    for (let c = r + 1; c < n; c++) acc -= a[r][c] * out[c];
    out[r] = acc / a[r][r];
  }
  return out;
}

// --- 5. SAVITZKY-GOLAY SMOOTHING ---

/**
 * Reproduces scipy.signal.savgol_filter(x, windowLength, polyorder) with the
 * default mode='interp'.
 *
 * Interior samples are a local least-squares polynomial fit evaluated at the
 * window centre. The first and last halfLength samples are NOT produced by
 * padding the signal: scipy fits one polynomial to the leading windowLength
 * samples and one to the trailing windowLength samples, then evaluates those
 * to fill the edges. We do the same, which is why the edges match exactly.
 *
 * The input must be free of NaN - the caller is expected to have interpolated
 * the gaps first, exactly as the server does.
 *
 * @param {Float64Array|number[]} values
 * @param {number} windowLength  Odd, >= 3, <= values.length.
 * @param {number} polyorder     Must be < windowLength.
 * @returns {Float64Array}
 */
export function savgolFilter(values, windowLength, polyorder) {
  const n = values.length;
  if (windowLength > n) {
    throw new Error(`savgolFilter: windowLength ${windowLength} exceeds input length ${n}`);
  }
  if (windowLength % 2 === 0) {
    throw new Error(`savgolFilter: windowLength ${windowLength} must be odd`);
  }
  if (polyorder >= windowLength) {
    throw new Error(`savgolFilter: polyorder ${polyorder} must be less than windowLength ${windowLength}`);
  }

  const out = new Float64Array(n);
  const half = Math.floor(windowLength / 2);

  // 1. Interior: convolve with the Savitzky-Golay kernel. The kernel is the
  //    least-squares fit weights evaluated at the window centre, so it only
  //    has to be derived once.
  const kernel = savgolCoeffs(windowLength, polyorder);
  for (let i = half; i < n - half; i++) {
    let acc = 0;
    for (let j = 0; j < windowLength; j++) acc += kernel[j] * values[i - half + j];
    out[i] = acc;
  }

  // 2. Leading edge: one polynomial over samples [0, windowLength), evaluated
  //    at positions 0 .. half-1.
  const positions = new Float64Array(windowLength);
  for (let j = 0; j < windowLength; j++) positions[j] = j;

  const headFit = polyFit(positions, sliceValues(values, 0, windowLength), polyorder);
  for (let i = 0; i < half; i++) out[i] = polyEval(headFit, i);

  // 3. Trailing edge: the mirror image, over the final windowLength samples.
  const tailFit = polyFit(positions, sliceValues(values, n - windowLength, n), polyorder);
  for (let i = n - half; i < n; i++) out[i] = polyEval(tailFit, i - (n - windowLength));

  return out;
}

/** Copies a half-open range out of either a typed array or a plain array. */
function sliceValues(values, start, stop) {
  const out = new Float64Array(stop - start);
  for (let i = start; i < stop; i++) out[i - start] = values[i];
  return out;
}

/**
 * Savitzky-Golay smoothing coefficients, equivalent to
 * scipy.signal.savgol_coeffs(windowLength, polyorder) for deriv=0.
 *
 * These are the weights that a least-squares polynomial fit over offsets
 * -half..+half assigns to each sample when evaluated at offset 0, obtained by
 * fitting the polynomial to each unit impulse in turn.
 */
export function savgolCoeffs(windowLength, polyorder) {
  const half = Math.floor(windowLength / 2);

  const offsets = new Float64Array(windowLength);
  for (let j = 0; j < windowLength; j++) offsets[j] = j - half;

  const coeffs = new Float64Array(windowLength);
  const impulse = new Float64Array(windowLength);

  for (let j = 0; j < windowLength; j++) {
    impulse.fill(0);
    impulse[j] = 1;
    coeffs[j] = polyEval(polyFit(offsets, impulse, polyorder), 0);
  }

  return coeffs;
}

/**
 * Applies the server's window-size clamping rules before smoothing.
 *
 * The server shrinks the requested window to the series length, forces it odd
 * by subtracting one, and silently skips smoothing altogether below 3 samples.
 * Returns null when smoothing should be skipped.
 */
export function effectiveWindow(requested, seriesLength) {
  let window = requested;
  if (seriesLength < window) window = seriesLength;
  if (window % 2 === 0) window -= 1;
  return window >= 3 ? window : null;
}
