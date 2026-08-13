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

export const DEFAULT_POINT_FITTER = "gaussianNonlinear";

/**
 * Minimum number of above-threshold samples required before attempting a fit.  A lone hot pixel,
 * or a couple of them, has no spatial extent to fit a Gaussian to.
 */
const MIN_SAMPLES = 14;

/**
 * Fraction of the peak height below which samples are excluded from the fit.
 */
const DEFAULT_RELATIVE_THRESHOLD = 0.2;

/**
 * Rescales patch samples in place to [0, 1], so the fitter below sees a fixed dynamic range
 * regardless of the source image's units or scale (uint8, uint16, or arbitrary float).  `NaN`
 * (missing) samples are left untouched.
 *
 * The fitter below fits a peak (a bright blob), not a trough. Set `invert` when the feature of
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
  // A flat or fully-missing patch: leave it as is, the fitter already rejects it.
  if (!(range > 0)) return;
  for (let i = 0; i < data.length; ++i) {
    data[i] = invert ? (max - data[i]) / range : (data[i] - min) / range;
  }
}

/**
 * Number of parameters in the full nonlinear Gaussian model: background, amplitude, and an
 * independent center and standard deviation per axis -- `[b, A, cx, cy, cz, sx, sy, sz]`.
 */
const NUM_PARAMS_NL = 8;

/** Levenberg-Marquardt iteration cap; the loop ordinarily converges in well under this. */
const MAX_ITERATIONS = 50;

/** Standard deviations are kept from collapsing to (or crossing) zero, which would be singular. */
const MIN_SIGMA = 0.25;

/**
 * Evaluates the Gaussian model and its residual and Jacobian contribution at one sample, and
 * accumulates them into the normal equations `JTJ` and `JTr` and the running sum-of-squares cost.
 *
 * This fits the intensity directly rather than its log, so no `height^2` reweighting is needed to
 * correct a log-transform bias -- ordinary least squares on the raw residual is already the right
 * objective.
 */
function accumulateGaussianResidual(
  p: Float64Array,
  x: number,
  y: number,
  z: number,
  value: number,
  JTJ: Float64Array,
  JTr: Float64Array,
  row: Float64Array,
): number {
  const [background, amplitude, cx, cy, cz, sx, sy, sz] = p;
  const u = x - cx;
  const v = y - cy;
  const w = z - cz;
  const g = Math.exp(
    -(
      (u * u) / (2 * sx * sx) +
      (v * v) / (2 * sy * sy) +
      (w * w) / (2 * sz * sz)
    ),
  );
  const model = background + amplitude * g;
  const residual = value - model;
  row[0] = 1;
  row[1] = g;
  row[2] = (amplitude * g * u) / (sx * sx);
  row[3] = (amplitude * g * v) / (sy * sy);
  row[4] = (amplitude * g * w) / (sz * sz);
  row[5] = (amplitude * g * u * u) / (sx * sx * sx);
  row[6] = (amplitude * g * v * v) / (sy * sy * sy);
  row[7] = (amplitude * g * w * w) / (sz * sz * sz);
  for (let a = 0; a < NUM_PARAMS_NL; ++a) {
    JTr[a] += row[a] * residual;
    for (let b = 0; b < NUM_PARAMS_NL; ++b) {
      JTJ[a * NUM_PARAMS_NL + b] += row[a] * row[b];
    }
  }
  return residual * residual;
}

/**
 * Accumulates the Gauss-Newton normal equations and sum-of-squares cost of the Gaussian model
 * `p` against every non-`NaN` sample in the patch.  Returns `undefined` if fewer than
 * `NUM_PARAMS_NL + 1` samples are usable, which leaves the system underdetermined.
 */
function evaluateModel(
  p: Float64Array,
  data: Float32Array,
  size: readonly [number, number, number],
): { JTJ: Float64Array; JTr: Float64Array; cost: number } | undefined {
  const JTJ = new Float64Array(NUM_PARAMS_NL * NUM_PARAMS_NL);
  const JTr = new Float64Array(NUM_PARAMS_NL);
  const row = new Float64Array(NUM_PARAMS_NL);
  let cost = 0;
  let count = 0;
  let i = 0;
  for (let z = 0; z < size[2]; ++z) {
    for (let y = 0; y < size[1]; ++y) {
      for (let x = 0; x < size[0]; ++x, ++i) {
        const value = data[i];
        if (Number.isNaN(value)) continue;
        cost += accumulateGaussianResidual(p, x, y, z, value, JTJ, JTr, row);
        ++count;
      }
    }
  }
  if (count < NUM_PARAMS_NL + 1) return undefined;
  return { JTJ, JTr, cost };
}

/**
 * Fits an axis-aligned 3-d Gaussian, `background + amplitude * exp(-sum((x - center)^2 /
 * (2*sigma^2)))`, by nonlinear least squares (Levenberg-Marquardt) against the raw patch
 * intensity.
 *
 * The background, amplitude, and per-axis standard deviation are fit alongside the center, rather
 * than derived from the patch, so the result isn't sensitive to a skewed background estimate. The
 * solve is seeded from the weighted mean and variance of the above-threshold samples -- a cheap
 * moment estimate that gets Levenberg-Marquardt close enough to converge in a handful of
 * iterations.
 *
 * Returns `undefined` if the initial estimate is degenerate, if too few samples are usable, if the
 * solve never improves on its seed, or if the converged fit fell to a trough or drifted outside
 * the sampled patch.
 */
export function gaussianNonlinearFitter(
  patch: VoxelPatch,
  options: { relativeThreshold?: number } = {},
): [number, number, number] | undefined {
  const { relativeThreshold = DEFAULT_RELATIVE_THRESHOLD } = options;
  const { data, size } = patch;
  let background = Number.POSITIVE_INFINITY;
  let peak = Number.NEGATIVE_INFINITY;
  for (const v of data) {
    if (v < background) background = v;
    if (v > peak) peak = v;
  }
  if (!Number.isFinite(background) || !Number.isFinite(peak)) return undefined;
  const amplitude = peak - background;
  if (!(amplitude > 0)) return undefined;
  const threshold = relativeThreshold * amplitude;

  let totalWeight = 0;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let aboveThreshold = 0;
  let i = 0;
  for (let z = 0; z < size[2]; ++z) {
    for (let y = 0; y < size[1]; ++y) {
      for (let x = 0; x < size[0]; ++x, ++i) {
        const height = data[i] - background;
        if (!(height > threshold)) continue;
        totalWeight += height;
        sumX += height * x;
        sumY += height * y;
        sumZ += height * z;
        ++aboveThreshold;
      }
    }
  }
  // A lone hot pixel, or a couple of them, has no spatial extent to fit a Gaussian to: the solve
  // would just shrink sigma to wrap it and report a spuriously confident center.
  if (aboveThreshold < MIN_SAMPLES) return undefined;
  const meanX = sumX / totalWeight;
  const meanY = sumY / totalWeight;
  const meanZ = sumZ / totalWeight;
  let varX = 0;
  let varY = 0;
  let varZ = 0;
  i = 0;
  for (let z = 0; z < size[2]; ++z) {
    for (let y = 0; y < size[1]; ++y) {
      for (let x = 0; x < size[0]; ++x, ++i) {
        const height = data[i] - background;
        if (!(height > threshold)) continue;
        varX += height * (x - meanX) ** 2;
        varY += height * (y - meanY) ** 2;
        varZ += height * (z - meanZ) ** 2;
      }
    }
  }

  let p = Float64Array.from([
    background,
    amplitude,
    meanX,
    meanY,
    meanZ,
    Math.max(Math.sqrt(varX / totalWeight), MIN_SIGMA),
    Math.max(Math.sqrt(varY / totalWeight), MIN_SIGMA),
    Math.max(Math.sqrt(varZ / totalWeight), MIN_SIGMA),
  ]);
  let state = evaluateModel(p, data, size);
  if (state === undefined) return undefined;

  // Levenberg-Marquardt: solve `(JTJ + lambda * diag(JTJ)) * delta = JTr` each iteration: large
  // when a step just made things worse (fall back toward gradient descent, which can't diverge),
  // small once steps are consistently improving (fall forward to the faster Gauss-Newton step).
  let lambda = 1e-3;
  for (let iter = 0; iter < MAX_ITERATIONS; ++iter) {
    const { JTJ, JTr, cost } = state;
    const damped = JTJ.slice();
    for (let a = 0; a < NUM_PARAMS_NL; ++a) {
      damped[a * NUM_PARAMS_NL + a] *= 1 + lambda;
    }
    const determinant = matrix.inverseInplace(damped, NUM_PARAMS_NL, NUM_PARAMS_NL);
    if (!Number.isFinite(determinant) || determinant === 0) return undefined;
    const delta = new Float64Array(NUM_PARAMS_NL);
    matrix.multiply(
      delta,
      NUM_PARAMS_NL,
      damped,
      NUM_PARAMS_NL,
      JTr,
      NUM_PARAMS_NL,
      NUM_PARAMS_NL,
      NUM_PARAMS_NL,
      1,
    );
    const candidate = Float64Array.from(p);
    for (let a = 0; a < NUM_PARAMS_NL; ++a) candidate[a] += delta[a];
    // Sigma is squared throughout the model, so its sign is unobservable; keep it positive and
    // away from zero rather than let the solve wander into a singular Jacobian.
    for (const a of [5, 6, 7]) {
      candidate[a] = Math.max(Math.abs(candidate[a]), MIN_SIGMA);
    }

    const candidateState = evaluateModel(candidate, data, size);
    if (candidateState !== undefined && candidateState.cost < cost) {
      const improved = (cost - candidateState.cost) / cost;
      p = candidate;
      state = candidateState;
      lambda = Math.max(lambda / 3, 1e-8);
      if (improved < 1e-9) break;
    } else {
      lambda *= 3;
      if (lambda > 1e8) break;
    }
  }

  // A negative amplitude means the solve slid toward a trough rather than a peak.
  if (!(p[1] > 0)) return undefined;
  const center: [number, number, number] = [p[2], p[3], p[4]];
  for (let k = 0; k < 3; ++k) {
    // Refuse to extrapolate a center outside the region that was actually sampled.
    if (!(center[k] >= 0 && center[k] <= size[k] - 1)) return undefined;
  }
  return center;
}

registerPointFitter("gaussianNonlinear", gaussianNonlinearFitter);
