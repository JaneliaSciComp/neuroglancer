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
  gaussianNonlinearFitter,
  getPointFitter,
  normalizePatch,
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

describe("gaussianNonlinearFitter", () => {
  it("is registered under the default name", () => {
    expect(getPointFitter(DEFAULT_POINT_FITTER)).toBe(gaussianNonlinearFitter);
  });

  it("recovers a sub-voxel center essentially exactly on a clean Gaussian", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const result = gaussianNonlinearFitter(makeGaussianPatch(center, 1.5));
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("handles an anisotropic Gaussian", () => {
    const center: [number, number, number] = [4.4, 5.9, 5.2];
    const result = gaussianNonlinearFitter(
      makeAnisotropicGaussianPatch(center, [1.2, 2.4, 1.8]),
    );
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("is unbiased when the blob is off-center in the patch", () => {
    const center: [number, number, number] = [3.1, 7.6, 5.0];
    const result = gaussianNonlinearFitter(makeGaussianPatch(center, 1.8));
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("recovers a sub-voxel center on a Gaussian at arbitrary float scale", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const patch = makeGaussianPatch(center, 1.5, {
      background: 2e-7,
      amplitude: 3.4e-6,
    });
    normalizePatch(patch);
    const result = gaussianNonlinearFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("recovers a dark blob's center once normalizePatch inverts it", () => {
    const center: [number, number, number] = [5.3, 4.7, 6.1];
    const patch = makeGaussianPatch(center, 1.5, {
      background: 220,
      amplitude: -200,
    });
    normalizePatch(patch, /* invert= */ true);
    const result = gaussianNonlinearFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("is robust to quantized, noisy samples", () => {
    // Fitting the raw intensity directly, rather than its log, avoids amplifying the noise of
    // dim samples.
    const center: [number, number, number] = [5.4, 5.15, 4.85];
    const patch = makeGaussianPatch(center, 2);
    for (let i = 0; i < patch.data.length; ++i) {
      const jitter = 4 * Math.sin(i * 12.9898) * Math.cos(i * 78.233);
      patch.data[i] = Math.round(patch.data[i] + jitter);
    }
    const result = gaussianNonlinearFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(0.01);
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
    const result = gaussianNonlinearFitter(patch);
    expect(result).toBeDefined();
    for (let i = 0; i < 3; ++i) {
      expect(Math.abs(result![i] - center[i])).toBeLessThan(1e-6);
    }
  });

  it("rejects a trough rather than reporting its center", () => {
    const patch = makeGaussianPatch([5, 5, 5], 2, {
      background: 220,
      amplitude: -200,
    });
    expect(gaussianNonlinearFitter(patch)).toBeUndefined();
  });

  it("returns undefined for a uniform patch", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(42);
    expect(
      gaussianNonlinearFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });

  it("returns undefined when every sample is missing", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(Number.NaN);
    expect(
      gaussianNonlinearFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });

  it("returns undefined when too few samples clear the threshold", () => {
    // A single bright voxel has no spatial extent to fit a Gaussian to.
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(10);
    data[5 + SIZE * (5 + SIZE * 5)] = 250;
    expect(
      gaussianNonlinearFitter({ data, size: [SIZE, SIZE, SIZE] }),
    ).toBeUndefined();
  });
});

describe("normalizePatch", () => {
  it("rescales an arbitrary-range float patch to [0, 1], preserving order", () => {
    const data = Float32Array.from([-1e6, -1, 0, 3.5, 1e6]);
    const patch = { data, size: [5, 1, 1] as [number, number, number] };
    normalizePatch(patch);
    expect(patch.data[0]).toBeCloseTo(0, 5);
    expect(patch.data[4]).toBeCloseTo(1, 5);
    for (let i = 1; i < patch.data.length; ++i) {
      expect(patch.data[i]).toBeGreaterThanOrEqual(patch.data[i - 1]);
    }
  });

  it("leaves NaN samples as NaN", () => {
    const data = Float32Array.from([0, Number.NaN, 10]);
    const patch = { data, size: [3, 1, 1] as [number, number, number] };
    normalizePatch(patch);
    expect(Number.isNaN(patch.data[1])).toBe(true);
  });

  it("is a no-op on a uniform patch", () => {
    const data = new Float32Array(SIZE * SIZE * SIZE).fill(42);
    const patch = { data, size: [SIZE, SIZE, SIZE] as [number, number, number] };
    normalizePatch(patch);
    expect(Array.from(patch.data)).toEqual(Array.from(data));
  });

  it("inverts, when requested, so the darkest sample maps to 1", () => {
    const data = Float32Array.from([-1e6, -1, 0, 3.5, 1e6]);
    const patch = { data, size: [5, 1, 1] as [number, number, number] };
    normalizePatch(patch, /* invert= */ true);
    expect(patch.data[0]).toBeCloseTo(1, 5);
    expect(patch.data[4]).toBeCloseTo(0, 5);
    for (let i = 1; i < patch.data.length; ++i) {
      expect(patch.data[i]).toBeLessThanOrEqual(patch.data[i - 1]);
    }
  });
});
