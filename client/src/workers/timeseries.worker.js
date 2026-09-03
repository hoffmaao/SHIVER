/**
 * timeseries.worker.js
 *
 * Runs the client-side timeseries extraction off the main thread.
 *
 * Extraction pulls ~15 MB Zarr chunks and then smooths tens of thousands of
 * samples. Doing that on the main thread would stall the map and the chart for
 * seconds at a time, so all of it happens here and only the finished payload
 * crosses back.
 *
 * Protocol - the main thread posts:
 *   { id, type: 'extract', roi, settings }
 *
 * and receives, for that id:
 *   { id, type: 'progress', progress }  zero or more times
 *   { id, type: 'partial', results }    velocities ready, error bars pending
 *   { id, type: 'result', results }     everything ready
 *   { id, type: 'error', message }      extraction could not run at all
 *
 * The worker keeps its Zarr chunk cache between messages, so repeated clicks
 * within the same 64x64 tile are served without touching the network.
 */

import { extractTimeseries } from '../utils/shiverTimeseries.js';

self.addEventListener('message', async (event) => {
  const { id, type, roi, settings } = event.data || {};

  if (type !== 'extract') return;

  try {
    const results = await extractTimeseries(roi, settings, {
      // Let the main thread draw velocities while the uncertainties download.
      onPartial: (partial) => self.postMessage({ id, type: 'partial', results: partial }),
      onProgress: (progress) => self.postMessage({ id, type: 'progress', progress }),
    });

    self.postMessage({ id, type: 'result', results });
  } catch (error) {
    // Only failures that prevented any extraction reach here; per-site problems
    // are reported inside the payload itself, exactly as the server does.
    self.postMessage({ id, type: 'error', message: error?.message || String(error) });
  }
});
