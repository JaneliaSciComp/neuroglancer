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

/**
 * @file The modal cross-section measurement tools.
 *
 * Arming a mode (ctrl+l for a line, ctrl+b for a box) activates one of these
 * tools.  While it is active, a left click sets the measurement's start point,
 * the end point follows the cursor, and a second left click finalizes it; the
 * mode stays armed so further measurements can be taken.  Escape deactivates
 * the tool, which clears the mode and the measurement.
 *
 * The tool owns only the *mode*: the arming state, the input event map that
 * takes the left button over from panning, and the status message.  Placing the
 * points is handled by `SliceViewPanel`, which is where the projection, bounds
 * clamping, edge auto-pan and labels already live.
 */

import type { MeasurementShape } from "#src/measurement_state.js";
import { StatusMessage } from "#src/status.js";
import type { ToolActivation } from "#src/ui/tool.js";
import { Tool } from "#src/ui/tool.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import type { Viewer } from "#src/viewer.js";

// `stopPropagation` is what keeps `translate-via-mouse-drag` (bound to a plain
// `at:mousedown0` in the slice view) from panning the view on the clicks that
// place the measurement.  The map is bound at `Number.POSITIVE_INFINITY`
// priority by `Viewer.toolInputEventMapBinder`, so it reliably wins.
const MEASURE_TOOL_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:mousedown0": { action: "measure-place-point", stopPropagation: true },
});

function describeShape(shape: MeasurementShape) {
  return shape === "line" ? "line" : "box";
}

export class MeasurementTool extends Tool<Viewer> {
  constructor(
    public viewer: Viewer,
    public shape: MeasurementShape,
  ) {
    super(viewer.toolBinder, /*toggle=*/ true);
  }

  get description() {
    return `measure ${describeShape(this.shape)}`;
  }

  toJSON() {
    return `measure${this.shape === "line" ? "Line" : "Box"}`;
  }

  activate(activation: ToolActivation<this>) {
    const { measurementState } = this.viewer;
    const shape = describeShape(this.shape);
    // A new mode always starts from a clean slate, so an old measurement is
    // never left behind under a different shape.
    measurementState.clear();
    measurementState.setActiveMeasurement(this.shape);
    activation.bindInputEventMap(MEASURE_TOOL_INPUT_EVENT_MAP);
    // Persistent (undelayed) message: the mode has no other visible indicator
    // once a measurement is finalized, so it stays up until the tool ends.
    const status = new StatusMessage(/*delay=*/ false);
    status.setText(
      `Measure ${shape}: click to set the start point, click again to ` +
        `finish, Escape to exit.`,
    );
    // Single teardown path, run on every exit route: escape
    // (deactivate-active-tool), arming the other mode, or viewer disposal.
    activation.registerDisposer(() => {
      status.dispose();
      measurementState.setActiveMeasurement(undefined);
      measurementState.clear();
    });
  }
}
