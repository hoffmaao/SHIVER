/**
 * timeseriesClient.js
 *
 * The single entry point MultiSourceMap.vue uses to get a timeseries.
 *
 * It prefers the client-side extraction, which reads Zarr straight from Source
 * Cooperative in a Web Worker and keeps the production VM out of the loop
 * entirely. If that path cannot run - an old browser, a blocked CDN, a decode
 * failure - it falls back to POST /api/timeseries/multi/json so the user still
 * gets their chart.
 *
 * Both paths return the identical payload shape, so nothing downstream has to
 * know which one answered. `source` on the result says which one did.
 */

import apiClient from '../api';

// --- 1. WORKER LIFECYCLE ---

let worker = null;
let workerUnavailable = false;
let nextRequestId = 1;

// Requests currently in flight, keyed by the id we gave the worker.
const inFlight = new Map();

/**
 * Returns the shared extraction worker, creating it on first use.
 *
 * Returns null when workers cannot be used at all: during the static build
 * (vite-ssg runs this module in Node, where Worker does not exist) or if
 * construction fails. Callers treat null as "use the server".
 */
function getWorker() {
  if (workerUnavailable) return null;
  if (worker) return worker;

  if (typeof window === 'undefined' || typeof Worker === 'undefined') {
    workerUnavailable = true;
    return null;
  }

  try {
    worker = new Worker(new URL('../workers/timeseries.worker.js', import.meta.url), {
      type: 'module',
    });

    worker.addEventListener('message', handleWorkerMessage);
    worker.addEventListener('error', handleWorkerFailure);

    return worker;
  } catch (error) {
    console.warn('SHIVER: extraction worker unavailable, using the server instead.', error);
    workerUnavailable = true;
    return null;
  }
}

function handleWorkerMessage(event) {
  const { id, type, results, progress, message } = event.data || {};

  const request = inFlight.get(id);
  if (!request) return;

  if (type === 'progress') { request.onProgress?.(progress); return; }
  if (type === 'partial') { request.onPartial?.(results); return; }

  inFlight.delete(id);
  if (type === 'result') request.resolve(results);
  else request.reject(new Error(message || 'Extraction failed in the worker.'));
}

/**
 * A worker-level error kills every in-flight request. We retire the worker so
 * the next call rebuilds it, and reject outstanding requests so they fall back
 * to the server rather than hanging.
 */
function handleWorkerFailure(event) {
  console.warn('SHIVER: extraction worker failed, falling back to the server.', event?.message || event);

  for (const [, request] of inFlight) {
    request.reject(new Error(event?.message || 'Extraction worker failed.'));
  }
  inFlight.clear();

  try { worker?.terminate(); } catch { /* already gone */ }
  worker = null;
}

// --- 2. THE TWO PATHS ---

/** Runs the extraction in the worker, resolving with the finished payload. */
function extractInWorker(activeWorker, roi, settings, hooks) {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++;
    inFlight.set(id, { resolve, reject, ...hooks });

    activeWorker.postMessage({ id, type: 'extract', roi, settings });
  });
}

/**
 * The original server request, kept as the fallback path.
 *
 * The auth header is still sent when we have a token, because the backend logs
 * interactions per user for the download statistics on the profile page.
 */
async function extractOnServer(roi, settings) {
  const token = typeof window !== 'undefined' ? sessionStorage.getItem('shiver_token') : null;
  const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};

  const response = await apiClient.post(
    '/api/timeseries/multi/json',
    { roi, ...settings },
    config,
  );
  return response.data;
}

// --- 3. PUBLIC API ---

/**
 * Extracts a timeseries for one or more map points.
 *
 * @param {Array<[number, number]>} roi  Locations as [latitude, longitude].
 * @param {object} settings              buffer, variable, sources, gap_fill,
 *                                       win_raw, win_daily, poly - the same
 *                                       fields the API accepts.
 * @param {object} [hooks]
 * @param {(results: object) => void} [hooks.onPartial]  Velocities are ready;
 *        error bars are still downloading. Client-side path only.
 * @param {(progress: object) => void} [hooks.onProgress]
 * @returns {Promise<{results: object, source: 'client'|'server'}>}
 */
export async function requestTimeseries(roi, settings, hooks = {}) {
  const activeWorker = getWorker();

  if (activeWorker) {
    try {
      const results = await extractInWorker(activeWorker, roi, settings, hooks);
      return { results, source: 'client' };
    } catch (error) {
      // Client-side extraction is an optimisation, never a hard dependency.
      console.warn('SHIVER: client-side extraction failed, retrying on the server.', error);
    }
  }

  return { results: await extractOnServer(roi, settings), source: 'server' };
}

/**
 * Releases the worker. Called when the map view unmounts so its Zarr chunk
 * cache - which can hold well over a hundred megabytes - does not outlive the
 * page that needed it.
 */
export function releaseTimeseriesWorker() {
  if (!worker) return;

  try { worker.terminate(); } catch { /* already gone */ }
  worker = null;
  inFlight.clear();
}
