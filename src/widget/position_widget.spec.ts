/**
 * @license
 * Copyright 2026 Google Inc.
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
import { makeCoordinateSpace } from "#src/coordinate_transform.js";
import {
  CoordinateDisplayMode,
  formatCoordinate,
  formatPosition,
  isTimeDimension,
} from "#src/widget/position_widget.js";

const { VOXEL, PHYSICAL } = CoordinateDisplayMode;

describe("formatCoordinate", () => {
  it("floors the voxel index in VOXEL mode", () => {
    expect(formatCoordinate(1024.7, 8e-9, "m", VOXEL)).toEqual("1024");
    expect(formatCoordinate(-0.5, 8e-9, "m", VOXEL)).toEqual("-1");
  });

  it("scales by the voxel size in PHYSICAL mode", () => {
    expect(formatCoordinate(1024, 8e-9, "m", PHYSICAL)).toEqual("8.192µm");
    expect(formatCoordinate(30, 8e-9, "m", PHYSICAL)).toEqual("240nm");
  });

  it("picks a prefix for negative coordinates", () => {
    // pickSiPrefix is undefined for negative input (Math.log10 gives NaN), so
    // the sign has to be handled separately.
    expect(formatCoordinate(-375, 8e-9, "m", PHYSICAL)).toEqual("-3µm");
  });

  it("anchors the prefix of a zero coordinate to the voxel size", () => {
    // Math.log10(0) is -Infinity; without special handling this yields "0ym".
    expect(formatCoordinate(0, 8e-9, "m", PHYSICAL)).toEqual("0nm");
    expect(formatCoordinate(0, 1e-3, "m", PHYSICAL)).toEqual("0mm");
  });

  it("does not elide a physical coordinate of exactly 1", () => {
    expect(formatCoordinate(1, 1e-9, "m", PHYSICAL)).toEqual("1nm");
  });

  it("shows only as many decimals as the voxel size resolves", () => {
    // 3303.5 in a Float32Array is exact, but 3303.5 * 8e-9 is 2.6427998e-5 in
    // float64, which at the default precision of 6 reads "26.427998µm".
    const p = Float32Array.of(3303.5);
    expect(formatCoordinate(p[0], 8e-9, "m", PHYSICAL)).toEqual("26.428µm");
    // A whole number of voxels keeps no spurious decimals.
    expect(formatCoordinate(30, 8e-9, "m", PHYSICAL)).toEqual("240nm");
    expect(formatCoordinate(3143, 8e-9, "m", PHYSICAL)).toEqual("25.144µm");
  });

  it("falls back to the voxel index for dimensions with no unit", () => {
    expect(formatCoordinate(1024.7, 1, "", PHYSICAL)).toEqual("1024");
  });
});

describe("formatPosition", () => {
  const space = makeCoordinateSpace({
    names: ["x", "y", "z"],
    scales: Float64Array.of(8e-9, 8e-9, 30e-9),
    units: ["m", "m", "m"],
  });

  it("labels each dimension in VOXEL mode", () => {
    expect(
      formatPosition(Float32Array.of(512, 1024.9, 30), space, VOXEL),
    ).toEqual("x 512  y 1024  z 30");
  });

  it("labels each dimension in PHYSICAL mode", () => {
    expect(
      formatPosition(Float32Array.of(512, 1024, 30), space, PHYSICAL),
    ).toEqual("x 4.096µm  y 8.192µm  z 900nm");
  });

  it("formats only the calibrated dimensions when units are mixed", () => {
    const mixed = makeCoordinateSpace({
      names: ["x", "c"],
      scales: Float64Array.of(8e-9, 1),
      units: ["m", ""],
    });
    expect(formatPosition(Float32Array.of(512, 3), mixed, PHYSICAL)).toEqual(
      "x 4.096µm  c 3",
    );
  });

  it("keeps time dimensions by default and drops them with omitTime", () => {
    const withTime = makeCoordinateSpace({
      names: ["t", "z", "y", "x"],
      scales: Float64Array.of(1e-3, 30e-9, 8e-9, 8e-9),
      units: ["s", "m", "m", "m"],
    });
    const p = Float32Array.of(7, 30, 1024, 512);
    // The cursor readout still shows everything.
    expect(formatPosition(p, withTime, VOXEL)).toEqual(
      "t 7  z 30  y 1024  x 512",
    );
    // Measurement labels omit the timepoint.
    expect(formatPosition(p, withTime, VOXEL, { omitTime: true })).toEqual(
      "z 30  y 1024  x 512",
    );
    expect(formatPosition(p, withTime, PHYSICAL, { omitTime: true })).toEqual(
      "z 900nm  y 8.192µm  x 4.096µm",
    );
  });

  it("drops an uncalibrated t axis with omitTime", () => {
    const untimed = makeCoordinateSpace({
      names: ["x", "y", "t"],
      scales: Float64Array.of(8e-9, 8e-9, 1),
      units: ["m", "m", ""],
    });
    expect(
      formatPosition(Float32Array.of(512, 1024, 3), untimed, VOXEL, {
        omitTime: true,
      }),
    ).toEqual("x 512  y 1024");
  });
});

describe("isTimeDimension", () => {
  it("recognizes any dimension calibrated in seconds", () => {
    // Every supported time unit normalizes to "s" with a scale.
    expect(isTimeDimension("t", "s")).toBe(true);
    expect(isTimeDimension("time", "s")).toBe(true);
  });

  it("recognizes an uncalibrated axis named t", () => {
    expect(isTimeDimension("t", "")).toBe(true);
    expect(isTimeDimension("t'", "")).toBe(true);
    expect(isTimeDimension("t^", "")).toBe(true);
  });

  it("does not treat spatial or channel dimensions as time", () => {
    expect(isTimeDimension("x", "m")).toBe(false);
    expect(isTimeDimension("z", "m")).toBe(false);
    expect(isTimeDimension("c'", "")).toBe(false);
    // A name merely starting with t is not a time axis.
    expect(isTimeDimension("theta", "")).toBe(false);
  });
});
