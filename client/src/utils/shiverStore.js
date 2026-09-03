/**
 * shiverStore.js
 *
 * Reads the SHIVER multi-source timeseries Zarr stores straight from Source
 * Cooperative, replacing the server-side xarray access in
 * server/utils/extract_multi_zarr_ts.py.
 *
 * The stores are public, anonymous and served with Access-Control-Allow-Origin,
 * so the browser can read them without the FastAPI backend being involved at
 * all. This module is what makes the timeseries extraction independent of the
 * single production VM.
 *
 * Two details of the store shape drive the design here:
 *
 *   1. The velocity arrays are chunked [time, 64, 64] - the WHOLE time axis
 *      sits in one chunk. That is ideal for a point timeseries (one request
 *      gets every epoch) but each chunk is ~15 MB compressed, so chunks are
 *      cached aggressively and shared between nearby points.
 *   2. data_source uses the Zarr v3 fixed_length_utf32 dtype, which zarrita
 *      cannot open. We read and decode that one array by hand.
 */

import * as zarr from 'zarrita';
import Zstd from 'numcodecs/zstd';
import proj4 from 'proj4';

const MS_PER_DAY = 86400000;

// --- 1. STORE CONFIGURATION ---

/**
 * Public Source Cooperative locations, with the polar stereographic
 * projection each region's grid is defined on. The proj4 strings are the same
 * ones MultiSourceMap.vue uses for its Leaflet CRS, and they agree with the
 * spatial_ref attributes inside the stores.
 */
export const REGION_STORES = {
  Greenland: {
    url: 'https://data.source.coop/uos-shiver/greenland/greenland_multisource_velocity_timeseries.zarr',
    code: 'EPSG:3413',
    proj: '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
  },
  Antarctica: {
    url: 'https://data.source.coop/uos-shiver/antarctica/antarctica_multisource_velocity_timeseries.zarr',
    code: 'EPSG:3031',
    proj: '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
  },
};

for (const { code, proj } of Object.values(REGION_STORES)) proj4.defs(code, proj);

/** Matches the server's region test: anything in the southern hemisphere is Antarctic. */
export function regionForLatitude(latitude) {
  return latitude < 0 ? 'Antarctica' : 'Greenland';
}

// --- 2. CHUNK CACHING ---

/**
 * A bounded, least-recently-used cache of compressed chunk bytes.
 *
 * Chunks are the expensive part of a client-side extraction - roughly 15 MB
 * over the network each - and adjacent map clicks very often land in the same
 * 64x64 (12.8 km) tile. Caching the compressed bytes rather than the decoded
 * floats keeps roughly seven times more chunks resident for the same memory,
 * and decoding is far cheaper than re-fetching.
 */
class ChunkCache {
  constructor(budgetBytes) {
    this.budgetBytes = budgetBytes;
    this.entries = new Map();      // insertion order doubles as LRU order
    this.usedBytes = 0;
  }

  get(key) {
    const value = this.entries.get(key);
    if (value === undefined) return undefined;

    // Re-insert so this key becomes the most recently used.
    this.entries.delete(key);
    this.entries.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.entries.has(key)) {
      this.usedBytes -= this.entries.get(key).byteLength;
      this.entries.delete(key);
    }
    this.entries.set(key, value);
    this.usedBytes += value.byteLength;

    // Evict from the front, which is the least recently used end.
    while (this.usedBytes > this.budgetBytes && this.entries.size > 1) {
      const oldest = this.entries.keys().next().value;
      this.usedBytes -= this.entries.get(oldest).byteLength;
      this.entries.delete(oldest);
    }
  }
}

/**
 * Wraps a zarrita store so chunk reads pass through the cache.
 *
 * Only keys that look like chunk data are cached; metadata documents are
 * small, and letting them through untouched keeps a store refresh cheap.
 */
function withChunkCache(store, cache) {
  return {
    async get(key, options) {
      const cached = cache.get(key);
      if (cached !== undefined) return cached;

      const value = await store.get(key, options);
      if (value && key.includes('/c/')) cache.set(key, value);
      return value;
    },
  };
}

// --- 3. OPENING A REGION ---

// One in-flight promise per region, so simultaneous point clicks on a cold
// cache share a single metadata fetch instead of racing.
const openRegions = new Map();

/**
 * Opens a region's store and reads its coordinate arrays.
 *
 * The coordinates (x, y, time, time_bnds, data_source) total well under a
 * megabyte and are needed by every extraction, so they are fetched once and
 * held for the lifetime of the page.
 *
 * @param {string} region 'Greenland' or 'Antarctica'.
 * @param {object} [options]
 * @param {number} [options.cacheBudgetBytes] Chunk cache size, default 192 MB.
 * @returns {Promise<object>} The opened region handle.
 */
export function openRegion(region, options = {}) {
  if (!openRegions.has(region)) {
    openRegions.set(region, loadRegion(region, options).catch((error) => {
      // Never cache a failure; the next attempt should be able to retry.
      openRegions.delete(region);
      throw error;
    }));
  }
  return openRegions.get(region);
}

async function loadRegion(region, options) {
  const config = REGION_STORES[region];
  if (!config) throw new Error(`Unknown region '${region}'`);

  const cache = new ChunkCache(options.cacheBudgetBytes ?? 192 * 1024 * 1024);
  const fetchStore = new zarr.FetchStore(config.url);
  const store = withChunkCache(fetchStore, cache);

  const group = await zarr.open(zarr.root(store), { kind: 'group' });

  // 1. Coordinates and the per-epoch metadata. These are single-chunk arrays.
  const [x, y, time, timeBnds] = await Promise.all(
    ['x', 'y', 'time', 'time_bnds'].map(async (name) => {
      const array = await zarr.open(group.resolve(name), { kind: 'array' });
      const chunk = await zarr.get(array, null);
      return { data: chunk.data, shape: array.shape, attrs: array.attrs };
    }),
  );

  // 2. Convert the time axis to milliseconds since the Unix epoch. The stores
  //    count from an unusual origin (1972-12-30), so the offset is read from
  //    the CF units attribute rather than assumed.
  const timeOrigin = parseTimeUnits(time.attrs?.units);
  const times = Array.from(time.data, (v) => Number(v) + timeOrigin);

  // 3. Acquisition pair separation in days, from the time bounds. The bounds
  //    use a different origin to the time axis, but the offset cancels in the
  //    difference so it does not need parsing.
  const timeSeparation = new Float64Array(times.length);
  for (let i = 0; i < times.length; i++) {
    const start = Number(timeBnds.data[i * 2]);
    const end = Number(timeBnds.data[i * 2 + 1]);
    timeSeparation[i] = (end - start) / MS_PER_DAY;
  }

  // 4. The contributing dataset per epoch, decoded by hand.
  const dataSource = await readFixedLengthUtf32(fetchStore, group, 'data_source');

  return {
    region,
    config,
    group,
    cache,
    xs: Array.from(x.data, Number),
    ys: Array.from(y.data, Number),
    times,
    timeSeparation,
    dataSource,
    attrs: group.attrs,
    // The store stamps its own version; surfacing it lets the UI say exactly
    // which vintage of the data a chart was drawn from.
    datasetVersion: group.attrs?.dataset_version ?? null,
    lastUpdated: group.attrs?.last_updated ?? null,
    arrays: new Map(),
  };
}

/**
 * Reads CF-style "milliseconds since YYYY-MM-DD" units and returns the origin
 * as milliseconds since the Unix epoch.
 */
function parseTimeUnits(units) {
  const match = /since\s+(\d{4})-(\d{2})-(\d{2})/.exec(units || '');
  if (!match) {
    throw new Error(`Could not read a time origin from units '${units}'`);
  }
  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

/**
 * Reads a Zarr v3 fixed_length_utf32 array.
 *
 * zarrita has no decoder for this dtype, so we fetch the single chunk, run it
 * through the same zstd codec zarrita would have used, and decode the
 * fixed-width UTF-32LE records ourselves. Records are null-padded to their
 * declared width.
 */
async function readFixedLengthUtf32(store, group, name) {
  const meta = await fetchArrayMetadata(store, name);
  const widthBytes = meta.data_type?.configuration?.length_bytes;
  if (!widthBytes) throw new Error(`Array '${name}' is not a fixed-width string array`);

  const compressed = await store.get(`/${name}/c/0`);
  if (!compressed) throw new Error(`Array '${name}' has no chunk to read`);

  const codec = Zstd.fromConfig({ id: 'zstd', level: 0 });
  const bytes = await codec.decode(compressed);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const charsPerRecord = widthBytes / 4;
  const count = meta.shape[0];
  const out = new Array(count);

  for (let i = 0; i < count; i++) {
    let text = '';
    for (let c = 0; c < charsPerRecord; c++) {
      const codePoint = view.getUint32(i * widthBytes + c * 4, true);
      if (codePoint === 0) break;                 // null padding ends the record
      text += String.fromCodePoint(codePoint);
    }
    out[i] = text;
  }
  return out;
}

/**
 * Pulls one array's metadata out of the group's consolidated metadata, falling
 * back to the array's own zarr.json if the store is not consolidated.
 */
async function fetchArrayMetadata(store, name) {
  const rootBytes = await store.get('/zarr.json');
  const root = JSON.parse(new TextDecoder().decode(rootBytes));

  const consolidated = root?.consolidated_metadata?.metadata?.[name];
  if (consolidated) return consolidated;

  const arrayBytes = await store.get(`/${name}/zarr.json`);
  if (!arrayBytes) throw new Error(`Array '${name}' not found in store`);
  return JSON.parse(new TextDecoder().decode(arrayBytes));
}

// --- 4. SITE SELECTION ---

/**
 * Turns a latitude/longitude and buffer into the grid selection the server
 * would have made.
 *
 * Mirrors the geometry handling in _process_single_site_multi: a zero buffer
 * takes the single nearest pixel, and any other buffer takes the square
 * bounding box of the buffered point, inclusive of both edges (which is what
 * an xarray label slice does).
 *
 * @returns {object} Either { error } or the resolved selection.
 */
export function resolveSite(regionInfo, latitude, longitude, buffer) {
  const { xs, ys } = regionInfo;

  // 1. Project into the store's polar stereographic grid.
  const [px, py] = proj4('EPSG:4326', regionInfo.config.code, [longitude, latitude]);

  // 2. Reject anything outside coverage before doing any work.
  const xMin = Math.min(xs[0], xs[xs.length - 1]);
  const xMax = Math.max(xs[0], xs[xs.length - 1]);
  const yMin = Math.min(ys[0], ys[ys.length - 1]);
  const yMax = Math.max(ys[0], ys[ys.length - 1]);

  if (px < xMin || px > xMax || py < yMin || py > yMax) {
    return { error: 'Location outside data coverage.' };
  }

  const centreX = nearestIndex(xs, px);
  const centreY = nearestIndex(ys, py);

  if (!(buffer > 0)) {
    return { singlePixel: true, ix: centreX, iy: centreY, px, py };
  }

  // 3. Inclusive label range in both axes. y descends, so its index range runs
  //    the opposite way to its coordinate range.
  const xRange = inclusiveRange(xs, px - buffer, px + buffer);
  const yRange = inclusiveRange(ys, py - buffer, py + buffer);

  // An empty window means the buffer fell between grid lines; the server falls
  // back to the single nearest pixel in that case.
  if (!xRange || !yRange) {
    return { singlePixel: true, ix: centreX, iy: centreY, px, py };
  }

  return { singlePixel: false, xRange, yRange, ix: centreX, iy: centreY, px, py };
}

/** Index of the coordinate closest to `target`, ties going to the lower index. */
function nearestIndex(coords, target) {
  let best = 0;
  let bestDistance = Infinity;

  for (let i = 0; i < coords.length; i++) {
    const distance = Math.abs(coords[i] - target);
    if (distance < bestDistance) { bestDistance = distance; best = i; }
  }
  return best;
}

/**
 * Index range [start, stop) of every coordinate lying within [lo, hi]
 * inclusive, for a coordinate array that may ascend or descend.
 * Returns null when nothing falls inside.
 */
function inclusiveRange(coords, lo, hi) {
  let start = -1;
  let stop = -1;

  for (let i = 0; i < coords.length; i++) {
    if (coords[i] >= lo && coords[i] <= hi) {
      if (start === -1) start = i;
      stop = i + 1;
    }
  }
  return start === -1 ? null : [start, stop];
}

// --- 5. READING A VARIABLE ---

/** Opens an array once per region and remembers the handle. */
async function getArray(regionInfo, name) {
  if (!regionInfo.arrays.has(name)) {
    regionInfo.arrays.set(
      name,
      zarr.open(regionInfo.group.resolve(name), { kind: 'array' }),
    );
  }
  return regionInfo.arrays.get(name);
}

/**
 * Reads one variable over the full time axis for a resolved site and
 * aggregates it spatially, exactly as the server does.
 *
 * A single-pixel selection is taken as-is. A windowed selection is reduced
 * with a per-epoch spatial median (ignoring gaps), which is what
 * subset.median(dim=['x','y']) computes.
 *
 * @returns {Promise<{values: Float64Array, validCount: Int32Array}>}
 *          validCount is the number of contributing pixels per epoch.
 */
export async function readVariable(regionInfo, name, site) {
  const array = await getArray(regionInfo, name);
  const epochs = regionInfo.times.length;

  // 1. Single pixel: one selection, no aggregation to do.
  if (site.singlePixel) {
    const chunk = await zarr.get(array, [null, site.iy, site.ix]);

    const values = Float64Array.from(chunk.data, Number);
    const validCount = new Int32Array(epochs);
    for (let t = 0; t < epochs; t++) validCount[t] = Number.isFinite(values[t]) ? 1 : 0;

    return { values, validCount };
  }

  // 2. Windowed: pull the cube and take a spatial median per epoch.
  const [y0, y1] = site.yRange;
  const [x0, x1] = site.xRange;

  const cube = await zarr.get(array, [null, zarr.slice(y0, y1), zarr.slice(x0, x1)]);
  const rows = y1 - y0;
  const cols = x1 - x0;
  const pixelsPerEpoch = rows * cols;

  const values = new Float64Array(epochs);
  const validCount = new Int32Array(epochs);
  const scratch = new Float64Array(pixelsPerEpoch);

  for (let t = 0; t < epochs; t++) {
    const base = t * pixelsPerEpoch;

    let n = 0;
    for (let p = 0; p < pixelsPerEpoch; p++) {
      const v = Number(cube.data[base + p]);
      if (Number.isFinite(v)) scratch[n++] = v;
    }

    validCount[t] = n;
    values[t] = n === 0 ? NaN : medianInPlace(scratch, n);
  }

  return { values, validCount };
}

/**
 * Median of the first `n` entries of `buffer`, matching numpy.nanmedian: for
 * an even count it averages the two central values. Sorts the buffer in place,
 * which is safe because the caller refills it each epoch.
 */
function medianInPlace(buffer, n) {
  const window = buffer.subarray(0, n);
  window.sort();

  const mid = n >> 1;
  return n % 2 === 1 ? window[mid] : (window[mid - 1] + window[mid]) / 2;
}
