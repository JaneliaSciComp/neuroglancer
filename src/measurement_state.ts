/**
 * @license
 * Copyright 2024 Google Inc.
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
 * @file Transient state for the cross-section measurement tool (a ruler drawn
 * with alt+right-drag).  The measurement is not part of the serialized viewer
 * state; it exists only until the user starts a new one or presses Escape.
 */

import type { CoordinateSpace } from "#src/coordinate_transform.js";
import { RefCounted } from "#src/util/disposable.js";
import type { quat } from "#src/util/geom.js";
import { quat as quatModule } from "#src/util/geom.js";
import { NullarySignal } from "#src/util/signal.js";

export type MeasurementShape = "line" | "box";

export class MeasurementState extends RefCounted {
  changed = new NullarySignal();
  shape: MeasurementShape = "line";
  // Endpoints in global (voxel) coordinates.
  startPosition: Float32Array | undefined = undefined;
  endPosition: Float32Array | undefined = undefined;
  // Coordinate space captured when the measurement was started, used to convert
  // the voxel distance to a physical length.
  coordinateSpace: CoordinateSpace | undefined = undefined;
  // Cross-section orientation the measurement was drawn in, so it is only shown
  // in panels displaying the same plane.
  orientation: quat = quatModule.create();
  // True while the user is actively dragging the second endpoint.
  active = false;

  get isSet() {
    return this.startPosition !== undefined && this.endPosition !== undefined;
  }

  begin(
    shape: MeasurementShape,
    position: Float32Array,
    orientation: quat,
    coordinateSpace: CoordinateSpace,
  ) {
    this.shape = shape;
    this.startPosition = Float32Array.from(position);
    this.endPosition = Float32Array.from(position);
    quatModule.copy(this.orientation, orientation);
    this.coordinateSpace = coordinateSpace;
    this.active = true;
    this.changed.dispatch();
  }

  update(position: Float32Array) {
    if (!this.active) return;
    this.endPosition = Float32Array.from(position);
    this.changed.dispatch();
  }

  finish() {
    if (!this.active) return;
    this.active = false;
    this.changed.dispatch();
  }

  clear() {
    if (
      this.startPosition === undefined &&
      this.endPosition === undefined &&
      !this.active
    ) {
      return;
    }
    this.startPosition = undefined;
    this.endPosition = undefined;
    this.coordinateSpace = undefined;
    this.active = false;
    this.changed.dispatch();
  }
}
