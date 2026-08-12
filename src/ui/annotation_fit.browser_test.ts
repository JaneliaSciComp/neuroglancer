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
import { gaussianLogFitter } from "#src/annotation/point_fit.js";
import type { ImageRenderLayer } from "#src/sliceview/volume/image_renderlayer.js";
import { samplePatch } from "#src/ui/annotation_fit.js";

const DISPLAY_DIMS = Int32Array.from([0, 1, 2]);

/**
 * Stands in for an image render layer.  Mirrors `VolumeChunkSource.getValueAt`, which floors the
 * continuous position to a voxel index: voxel `i` covers `[i, i+1)` and is centered at `i + 0.5`.
 */
function makeStubLayer(
  valueForVoxel: (i: number, j: number, k: number) => number | null,
) {
  return {
    getValueAt(position: Float32Array) {
      return valueForVoxel(
        Math.floor(position[0]),
        Math.floor(position[1]),
        Math.floor(position[2]),
      );
    },
  } as unknown as ImageRenderLayer;
}

describe("samplePatch", () => {
  it("samples voxel centers, so origin is half-integer", () => {
    const layer = makeStubLayer((i, j, k) => i + 100 * j + 10000 * k);
    const sampled = samplePatch(
      layer,
      Float32Array.from([20.9, 30.2, 40.5]),
      DISPLAY_DIMS,
      2,
    )!;
    expect(sampled).toBeDefined();
    // The click falls inside voxel [20, 30, 40], so the patch spans voxels 18..22 etc.
    expect(sampled.origin).toEqual([18.5, 28.5, 38.5]);
    expect(Array.from(sampled.patch.size)).toEqual([5, 5, 5]);
    // Sample [0,0,0] must be voxel [18, 28, 38], not [19, 29, 39] or [17, 27, 37].
    expect(sampled.patch.data[0]).toBe(18 + 100 * 28 + 10000 * 38);
    // ...and the last sample voxel [22, 32, 42].
    expect(sampled.patch.data[5 * 5 * 5 - 1]).toBe(22 + 100 * 32 + 10000 * 42);
  });

  it("recovers the true continuous center of a Gaussian, with no half-voxel bias", () => {
    // Ground truth in continuous global coordinates.
    const center = [16.3, 20.7, 24.2];
    const sigma = 2.5;
    // Voxel `i` holds the Gaussian evaluated at its center, `i + 0.5`.
    const layer = makeStubLayer((i, j, k) => {
      if (i < 0 || j < 0 || k < 0 || i >= 64 || j >= 64 || k >= 64) return null;
      const d2 =
        (i + 0.5 - center[0]) ** 2 +
        (j + 0.5 - center[1]) ** 2 +
        (k + 0.5 - center[2]) ** 2;
      return 8 + 230 * Math.exp(-d2 / (2 * sigma * sigma));
    });
    // Click deliberately off-center.
    const sampled = samplePatch(
      layer,
      Float32Array.from([17.75, 22, 24]),
      DISPLAY_DIMS,
      5,
    )!;
    const fit = gaussianLogFitter(sampled.patch)!;
    expect(fit).toBeDefined();
    for (let k = 0; k < 3; ++k) {
      // Bounded by the float32 rounding of the sampled values, not by the fit; a half-voxel
      // convention error would show up here as ~0.5.
      expect(Math.abs(sampled.origin[k] + fit[k] - center[k])).toBeLessThan(
        1e-4,
      );
    }
  });

  it("returns undefined when most samples are unavailable", () => {
    const layer = makeStubLayer(() => null);
    expect(
      samplePatch(layer, Float32Array.from([10, 10, 10]), DISPLAY_DIMS, 3),
    ).toBeUndefined();
  });

  it("returns undefined when the display space has rank below 3", () => {
    const layer = makeStubLayer(() => 1);
    expect(
      samplePatch(
        layer,
        Float32Array.from([10, 10, 10]),
        Int32Array.from([0, 1, -1]),
        3,
      ),
    ).toBeUndefined();
  });
});
