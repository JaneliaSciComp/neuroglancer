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

import { describe, expect, it } from "vitest";
import type { VoxelPatch } from "#src/annotation/point_fit.js";
import {
  DEFAULT_POINT_FITTER,
  centroidFitter,
  gaussianLogFitter,
  getPointFitter,
} from "#src/annotation/point_fit.js";

const SIZE = 11;

function makeGaussianPatch(
  center: readonly [number, number, number],
  sigma: number,
  options: { background?: number; amplitude?: number } = {},
): VoxelPatch {
  const { background = 20, amplitude = 200 } = options;
  const data = new Float32Array(SIZE * SIZE * SIZE);
  let i = 0;
  for (let z = 0; z < SIZE; ++z) {
    for (let y = 0; y < SIZE; ++y) {
      for (let x = 0; x < SIZE; ++x, ++i) {
        const d2 =
          (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2;
        data[i] = background + amplitude * Math.exp(-d2 / (2 * sigma * sigma));
      }
    }
  }
  return { data, size: [SIZE, SIZE, SIZE] };
}

function makeAnisotropicGaussianPatch(
  center: readonly [number, number, number],
  sigma: readonly [number, number, number],
): VoxelPatch {
  const data = new Float32Array(SIZE * SIZE * SIZE);
  let i = 0;
  for (let z = 0; z < SIZE; ++z) {
    for (let y = 0; y < SIZE; ++y) {
      for (let x = 0; x < SIZE; ++x, ++i) {
        const e =
          (x - center[0]) ** 2 / (2 * sigma[0] ** 2) +
          (y - center[1]) ** 2 / (2 * sigma[1] ** 2) +
          (z - center[2]) ** 2 / (2 * sigma[2] ** 2);
        data[i] = 20 + 200 * Math.exp(-e);
      }
    }
  }
  return { data, size: [SIZE, SIZE, SIZE] };
}

describe("gaussianLogFitter", () => {
  it("is registered under the default name", () => {
    expect(getPointFitter(DEFAULT_POINT_FITTER)).toBe(gaussianLogFitter);
  });

  it("recovers a sub-voxel center essentially exactly on a clean Gaussian", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const result = gaussianLogFitter(makeGaussianPatch(center, 1.5));
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("handles an anisotropic Gaussian", () => {
    const center: [number, number, number] = [4.4, 5.9, 5.2];
    const result = gaussianLogFitter(
      makeAnisotropicGaussianPatch(center, [1.2, 2.4, 1.8]),
    );
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("beats the centroid when the blob is off-center in the patch", () => {
    // The window truncates the blob's tails asymmetrically, which biases a centroid but not a fit
    // restricted to the peak core.
    const center: [number, number, number] = [3.1, 7.6, 5.0];
    const patch = makeGaussianPatch(center, 1.8);
    const fitted = gaussianLogFitter(patch)!;
    const centroid = centroidFitter(patch)!;
    const error = (p: number[]) =>
      Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    expect(error(fitted)).toBeLessThan(1e-6);
    expect(error(fitted)).toBeLessThan(error(centroid));
  });

  it("is robust to quantized, noisy samples", () => {
    const center: [number, number, number] = [5.4, 5.15, 4.85];
    const patch = makeGaussianPatch(center, 2);
    // Deterministic pseudo-noise, then quantized to integers as a uint8 image would be.
    for (let i = 0; i < patch.data.length; ++i) {
      const jitter = 4 * Math.sin(i * 12.9898) * Math.cos(i * 78.233);
      patch.data[i] = Math.round(patch.data[i] + jitter);
    }
    const result = gaussianLogFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(0.1);
    }
  });

  it("rejects a trough rather than reporting its center", () => {
    const patch = makeGaussianPatch([5, 5, 5], 2, {
      background: 220,
      amplitude: -200,
    });
    expect(gaussianLogFitter(patch)).toBeUndefined();
  });

  it("returns undefined for a uniform patch", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(42);
    expect(
      gaussianLogFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });

  it("returns undefined when every sample is missing", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(Number.NaN);
    expect(
      gaussianLogFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });

  it("returns undefined when too few samples clear the threshold", () => {
    // A single bright voxel gives no curvature to fit.
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(10);
    data[5 + SIZE * (5 + SIZE * 5)] = 250;
    expect(
      gaussianLogFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });
});

describe("centroidFitter", () => {
  it("is registered under its own name", () => {
    expect(getPointFitter("centroid")).toBe(centroidFitter);
  });

  it("recovers a sub-voxel Gaussian center", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const result = centroidFitter(makeGaussianPatch(center, 1.5));
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(0.15);
    }
  });

  it("ignores missing samples", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const patch = makeGaussianPatch(center, 1.5);
    // Blank out a corner far from the blob; the fit must not be dragged by it.
    for (let z = 0; z < 2; ++z) {
      for (let y = 0; y < 2; ++y) {
        for (let x = 0; x < 2; ++x) {
          patch.data[x + SIZE * (y + SIZE * z)] = Number.NaN;
        }
      }
    }
    const result = centroidFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(0.15);
    }
  });

  it("returns undefined for a uniform patch", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(42);
    expect(centroidFitter({ data, size: [SIZE, SIZE, SIZE] })).toBeUndefined();
  });

  it("returns undefined when every sample is missing", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(Number.NaN);
    expect(centroidFitter({ data, size: [SIZE, SIZE, SIZE] })).toBeUndefined();
  });
});
