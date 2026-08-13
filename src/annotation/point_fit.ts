/**
 * @license
 * Copyright 2026 Howard Hughes Medical Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Sub-voxel center estimation from a small 3-d patch of image intensity.
 *
 * This module depends only on `#src/util/matrix.js`, itself a dependency-free numerics helper: it
 * is the seam at which a different fitting implementation can be substituted.  To use your own,
 * write a function matching `PointFitter` and call `registerPointFitter`; the annotation tool
 * selects a fitter by name from its JSON state.
 */

import * as matrix from "#src/util/matrix.js";

export interface VoxelPatch {
  /**
   * Sample values, with x varying fastest: `index = x + size[0] * (y + size[1] * z)`.  A value of
   * `NaN` marks a sample that could not be read, because the containing chunk was not resident or
   * the position was outside the volume.
   */
  data: Float32Array;
  size: readonly [number, number, number];
}

/**
 * Returns the fitted center as fractional patch indices, or `undefined` if no center could be
 * determined.
 */
export type PointFitter = (
  patch: VoxelPatch,
) => [number, number, number] | undefined;

const pointFitters = new Map<string, PointFitter>();

export function registerPointFitter(name: string, fitter: PointFitter) {
  pointFitters.set(name, fitter);
}

export function getPointFitter(name: string): PointFitter | undefined {
  return pointFitters.get(name);
}

export function getPointFitterNames(): string[] {
  return Array.from(pointFitters.keys());
}

export const DEFAULT_POINT_FITTER = "gaussianLog";

/**
 * Number of coefficients in the quadratic log-intensity model: `[1, u, u^2, v, v^2, w, w^2]`.
 */
const NUM_PARAMS = 7;

/**
 * Minimum number of above-threshold samples required before attempting the solve.  Seven is the
 * bare rank requirement; twice that gives the fit something to average over.
 */
const MIN_SAMPLES = 2 * NUM_PARAMS;

/**
 * Fraction of the peak height below which samples are excluded from the fit.
 */
const DEFAULT_RELATIVE_THRESHOLD = 0.2;

/**
 * Rescales patch samples in place to [0, 1], so the fitters below see a fixed dynamic range
 * regardless of the source image's units or scale (uint8, uint16, or arbitrary float).  `NaN`
 * (missing) samples are left untouched.
 *
 * Both fitters below fit a peak (a bright blob), not a trough. Set `invert` when the feature of
 * interest is dark on a bright background, so the fitted blob is the dark spot rather than its
 * bright surroundings.
 */
export function normalizePatch(patch: VoxelPatch, invert = false): void {
  const { data } = patch;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const v of data) {
    // Both comparisons are false for NaN, so missing samples never become the extremes.
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  // A flat or fully-missing patch: leave it as is, the fitters already reject it.
  if (!(range > 0)) return;
  for (let i = 0; i < data.length; ++i) {
    data[i] = invert ? (max - data[i]) / range : (data[i] - min) / range;
  }
}

/**
 * Fits an axis-aligned 3-d Gaussian by linear least squares on the log-transformed intensity.
 *
 * Taking the log of a Gaussian yields a quadratic with no cross terms,
 *
 *     ln(I - background) = c0 + c1*u + c2*u^2 + c3*v + c4*v^2 + c5*w + c6*w^2
 *
 * which is linear in its coefficients, so the fit is a single closed-form solve rather than an
 * iterative optimization.  The center along each axis is the vertex of the corresponding parabola,
 * `-c1 / (2*c2)`, and downward curvature (`c2 < 0`) is what distinguishes a peak from a saddle or
 * a trough.
 *
 * Two details make this behave on real data.  The background is taken to be the patch minimum and
 * only samples above `relativeThreshold` of the peak height are fit, which confines the model to
 * the core of the blob where it actually holds and keeps the result from being dragged by a
 * truncated tail.  Samples are weighted by the squared background-subtracted height, which
 * compensates for the log transform inflating the variance of dim samples by `1 / height^2`; this
 * is the standard correction to the naive log-parabola fit, which otherwise lets the noisiest
 * samples dominate.
 *
 * Returns `undefined` if there are too few usable samples, if the system is singular, if any axis
 * lacks downward curvature, or if the resulting center falls outside the sampled patch.
 */
export function gaussianLogFitter(
  patch: VoxelPatch,
  options: { relativeThreshold?: number } = {},
): [number, number, number] | undefined {
  const { relativeThreshold = DEFAULT_RELATIVE_THRESHOLD } = options;
  const { data, size } = patch;
  let background = Number.POSITIVE_INFINITY;
  let peak = Number.NEGATIVE_INFINITY;
  for (const v of data) {
    // Both comparisons are false for NaN, so missing samples never become the extremes.
    if (v < background) background = v;
    if (v > peak) peak = v;
  }
  if (!Number.isFinite(background) || !Number.isFinite(peak)) return undefined;
  const threshold = relativeThreshold * (peak - background);
  if (!(threshold > 0)) return undefined;

  // Centering the coordinates on the patch keeps the normal equations well conditioned.
  const origin = [(size[0] - 1) / 2, (size[1] - 1) / 2, (size[2] - 1) / 2];

  // Normal equations `ata * coefficients = atb`, column-major.  `ata` is symmetric, so the
  // column-major/row-major distinction does not matter for it.
  const ata = new Float64Array(NUM_PARAMS * NUM_PARAMS);
  const atb = new Float64Array(NUM_PARAMS);
  const row = new Float64Array(NUM_PARAMS);
  row[0] = 1;
  let count = 0;
  let i = 0;
  for (let z = 0; z < size[2]; ++z) {
    const w = z - origin[2];
    for (let y = 0; y < size[1]; ++y) {
      const v = y - origin[1];
      for (let x = 0; x < size[0]; ++x, ++i) {
        const height = data[i] - background;
        // False for NaN, and for everything at or below the threshold.
        if (!(height > threshold)) continue;
        const u = x - origin[0];
        row[1] = u;
        row[2] = u * u;
        row[3] = v;
        row[4] = v * v;
        row[5] = w;
        row[6] = w * w;
        const weight = height * height;
        const value = Math.log(height);
        for (let a = 0; a < NUM_PARAMS; ++a) {
          const weighted = weight * row[a];
          atb[a] += weighted * value;
          for (let b = 0; b < NUM_PARAMS; ++b) {
            ata[a * NUM_PARAMS + b] += weighted * row[b];
          }
        }
        ++count;
      }
    }
  }
  if (count < MIN_SAMPLES) return undefined;

  const determinant = matrix.inverseInplace(ata, NUM_PARAMS, NUM_PARAMS);
  if (!Number.isFinite(determinant) || determinant === 0) return undefined;
  const coefficients = new Float64Array(NUM_PARAMS);
  matrix.multiply(
    coefficients,
    NUM_PARAMS,
    ata,
    NUM_PARAMS,
    atb,
    NUM_PARAMS,
    NUM_PARAMS,
    NUM_PARAMS,
    1,
  );

  const center: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; ++k) {
    const linear = coefficients[1 + 2 * k];
    const quadratic = coefficients[2 + 2 * k];
    // A maximum requires downward curvature; anything else is not a peak.
    if (!(quadratic < 0)) return undefined;
    const position = origin[k] - linear / (2 * quadratic);
    // Refuse to extrapolate a center outside the region that was actually sampled.
    if (!(position >= 0 && position <= size[k] - 1)) return undefined;
    center[k] = position;
  }
  return center;
}

/**
 * Background-subtracted intensity-weighted center of mass.  The background is taken to be the
 * minimum sample in the patch, so a patch containing a single bright blob on a flat background
 * yields that blob's centroid, which coincides with the center of a symmetric Gaussian.
 *
 * Cheaper and more robust than `gaussianLogFitter` on noisy or non-Gaussian blobs, but biased
 * whenever the patch is not roughly centered on the blob, since the window then truncates the
 * tails asymmetrically.  Kept as a fallback; select it with the tool's `fitter` option.
 */
export function centroidFitter(
  patch: VoxelPatch,
): [number, number, number] | undefined {
  const { data, size } = patch;
  let background = Number.POSITIVE_INFINITY;
  for (const v of data) {
    // Comparison is false for NaN, so missing samples are skipped.
    if (v < background) background = v;
  }
  if (!Number.isFinite(background)) return undefined;
  let totalWeight = 0;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let i = 0;
  for (let z = 0; z < size[2]; ++z) {
    for (let y = 0; y < size[1]; ++y) {
      for (let x = 0; x < size[0]; ++x, ++i) {
        const weight = data[i] - background;
        // False for NaN and for zero-weight samples.
        if (!(weight > 0)) continue;
        totalWeight += weight;
        sumX += weight * x;
        sumY += weight * y;
        sumZ += weight * z;
      }
    }
  }
  if (totalWeight === 0) return undefined;
  return [sumX / totalWeight, sumY / totalWeight, sumZ / totalWeight];
}

registerPointFitter("gaussianLog", gaussianLogFitter);
registerPointFitter("centroid", centroidFitter);
