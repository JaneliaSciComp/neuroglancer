/**
 * @license
 * Copyright 2016 Google Inc.
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

import { AxesLineHelper, computeAxisLineMatrix } from "#src/axes_lines.js";
import type { CoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import {
  computeHoverMarkerMatrix,
  crossSectionHoverMarkerAlpha,
  shouldDrawCrossSectionHoverMarker,
} from "#src/hover_position_marker.js";
import { HoverPositionMarker } from "#src/hover_position_marker_renderer.js";
import type { VisibleRenderLayerTracker } from "#src/layer/index.js";
import { makeRenderedPanelVisibleLayerTracker } from "#src/layer/index.js";
import type { NdcPoint } from "#src/measurement_line_render_helper.js";
import { MeasurementLineRenderHelper } from "#src/measurement_line_render_helper.js";
import type { MeasurementShape } from "#src/measurement_state.js";
import { PickIDManager } from "#src/object_picking.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type {
  FramePickingData,
  RenderedDataViewerState,
} from "#src/rendered_data_panel.js";
import { RenderedDataPanel } from "#src/rendered_data_panel.js";
import {
  getPickDiameter,
  getPickOffsetSequence,
  resolveNearestPanelPickSample,
} from "#src/rendered_data_panel_picking.js";
import type { SliceView } from "#src/sliceview/frontend.js";
import { SliceViewRenderHelper } from "#src/sliceview/frontend.js";
import type {
  SliceViewPanelReadyRenderContext,
  SliceViewPanelRenderContext,
} from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import type { TrackableBoolean } from "#src/trackable_boolean.js";
import type { TrackableRGB } from "#src/util/color.js";
import type { Borrowed, Owned } from "#src/util/disposable.js";
import { removeFromParent } from "#src/util/dom.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import { registerActionListener } from "#src/util/event_action_map.js";
import {
  disableZProjection,
  identityMat4,
  kAxes,
  mat4,
  vec3,
  vec4,
} from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";
import { formatScaleWithUnitAsString } from "#src/util/si_units.js";
import type { TouchRotateInfo } from "#src/util/touch_bindings.js";
import {
  FramebufferConfiguration,
  OffscreenCopyHelper,
  TextureBuffer,
} from "#src/webgl/offscreen.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";
import {
  CoordinateDisplayMode,
  formatPosition,
} from "#src/widget/position_widget.js";
import type { TrackableScaleBarOptions } from "#src/widget/scale_bar.js";
import { MultipleScaleBarTextures } from "#src/widget/scale_bar.js";

export interface SliceViewerState extends RenderedDataViewerState {
  showScaleBar: TrackableBoolean;
  wireFrame: TrackableBoolean;
  scaleBarOptions: TrackableScaleBarOptions;
  crossSectionBackgroundColor: TrackableRGB;
  hideCrossSectionBackground3D: TrackableBoolean;
}

export enum OffscreenTextures {
  COLOR = 0,
  PICK = 1,
  NUM_TEXTURES = 2,
}

function sliceViewPanelEmitColorAndPickID(builder: ShaderBuilder) {
  builder.addOutputBuffer("vec4", "out_fragColor", 0);
  builder.addOutputBuffer("highp vec4", "out_pickId", 1);
  builder.addFragmentCode(`
void emit(vec4 color, highp uint pickId) {
  out_fragColor = color;
  float pickIdFloat = float(pickId);
  out_pickId = vec4(pickIdFloat, pickIdFloat, pickIdFloat, 1.0);
}
`);
}

function sliceViewPanelEmitColor(builder: ShaderBuilder) {
  builder.addOutputBuffer("vec4", "out_fragColor", null);
  builder.addFragmentCode(`
void emit(vec4 color, highp uint pickId) {
  out_fragColor = color;
}
`);
}

const tempVec3 = vec3.create();
const tempVec3b = vec3.create();
const tempVec3c = vec3.create();
const tempVec3d = vec3.create();
const tempVec4 = vec4.create();
const tempHoverColor = vec4.create();

// Half-length, in logical pixels, of the "+" reticle drawn at the hover
// position that is synchronized across cross-section panels.
const HOVER_MARKER_SIZE = 8;
const tempVec4b = vec4.create();

// Width in element pixels of the band along each panel edge that triggers
// auto-panning while a measurement is being drawn.
const MEASURE_EDGE_PAN_MARGIN = 40;
// Maximum auto-pan speed, in viewport pixels per animation frame, reached once
// the cursor sits on (or past) a panel edge.  Kept deliberately slow so the view
// does not race out of the visible volume before the user can react.
const MEASURE_EDGE_PAN_MAX_SPEED = 3;
// How far a panel's cross-section plane may be tilted away from the plane a
// measurement was drawn in and still show it.  A purely in-plane rotation leaves
// the plane normal unchanged and is therefore always allowed; this tolerance
// only governs out-of-plane tilt.
const MEASURE_PLANE_MATCH_TOLERANCE_DEGREES = 5;
const MEASURE_PLANE_MATCH_MIN_COS = Math.cos(
  (MEASURE_PLANE_MATCH_TOLERANCE_DEGREES * Math.PI) / 180,
);

// Given the cursor position (in element pixels) and the panel size, returns the
// [deltaX, deltaY] to feed to `translateByViewportPixels` so the view scrolls to
// reveal content past whichever edge the cursor has reached.  Returns [0, 0]
// when the cursor is comfortably inside the panel.  The speed ramps up with how
// deeply the cursor has entered the edge band and saturates at the panel edge.
function computeEdgePanVelocity(
  mouseX: number,
  mouseY: number,
  width: number,
  height: number,
): [number, number] {
  const axis = (pos: number, size: number): number => {
    if (pos < MEASURE_EDGE_PAN_MARGIN) {
      // Near the low edge: reveal content before it, so content slides toward
      // the positive direction.
      const depth = Math.min(
        MEASURE_EDGE_PAN_MARGIN - pos,
        MEASURE_EDGE_PAN_MARGIN,
      );
      return (depth / MEASURE_EDGE_PAN_MARGIN) * MEASURE_EDGE_PAN_MAX_SPEED;
    }
    if (pos > size - MEASURE_EDGE_PAN_MARGIN) {
      const depth = Math.min(
        pos - (size - MEASURE_EDGE_PAN_MARGIN),
        MEASURE_EDGE_PAN_MARGIN,
      );
      return -(depth / MEASURE_EDGE_PAN_MARGIN) * MEASURE_EDGE_PAN_MAX_SPEED;
    }
    return 0;
  };
  return [axis(mouseX, width), axis(mouseY, height)];
}

// Clamps `position` into the dataset bounding box (per global dimension) writing
// the result to `out`, and returns whether `position` was already inside the box
// (i.e. no clamping was needed).  Unbounded dimensions (±Infinity) never clamp.
// Used to stop a measurement from being drawn past the edge of the volume: the
// endpoint sticks to the boundary once the cursor moves outside the data.
function clampPositionToBounds(
  out: Float32Array,
  position: Float32Array,
  coordinateSpace: CoordinateSpace,
): boolean {
  const { lowerBounds, upperBounds } = coordinateSpace.bounds;
  let inBounds = true;
  const rank = position.length;
  for (let i = 0; i < rank; ++i) {
    const v = position[i];
    const lo = lowerBounds[i];
    const hi = upperBounds[i];
    if (v < lo) {
      out[i] = lo;
      inBounds = false;
    } else if (v > hi) {
      out[i] = hi;
      inBounds = false;
    } else {
      out[i] = v;
    }
  }
  return inBounds;
}

// Formats the length of the measurement segment between two global positions,
// following the cursor coordinate-display mode so the ruler and the cursor
// readout always agree.  In PHYSICAL mode the per-dimension voxel deltas are
// scaled before being combined; the length is reported in the first calibrated
// dimension's unit.  A coordinate space with no units at all falls back to
// voxels regardless of the mode.
function formatMeasurementLength(
  start: Float32Array,
  end: Float32Array,
  coordinateSpace: CoordinateSpace,
  mode: CoordinateDisplayMode,
): string {
  const { rank, scales, units } = coordinateSpace;
  let voxelSq = 0;
  let physicalSq = 0;
  let unit = "";
  for (let i = 0; i < rank; ++i) {
    const d = end[i] - start[i];
    voxelSq += d * d;
    const p = d * scales[i];
    physicalSq += p * p;
    if (unit === "" && units[i] !== "") unit = units[i];
  }
  if (mode === CoordinateDisplayMode.PHYSICAL && unit !== "") {
    return formatScaleWithUnitAsString(Math.sqrt(physicalSq), unit, {
      elide1: false,
      precision: 4,
    });
  }
  return `${Math.sqrt(voxelSq).toFixed(1)} vox`;
}

// Endpoint coordinates are formatted with `formatPosition` from the position
// widget, so each axis is labeled with its dimension name in the natural
// coordinate-space order and honors the same voxel/physical mode as the cursor
// readout in the top bar.

export class SliceViewPanel extends RenderedDataPanel {
  declare viewer: SliceViewerState;
  private sliceViewRenderHelper;

  private axesLineHelper = this.registerDisposer(AxesLineHelper.get(this.gl));
  private hoverMarker = this.registerDisposer(HoverPositionMarker.get(this.gl));
  private measurementLineHelper = this.registerDisposer(
    MeasurementLineRenderHelper.get(this.gl),
  );
  private measurementColor = vec4.fromValues(0, 1, 1, 1); // cyan
  // Length label at the midpoint, plus a coordinate label next to each endpoint.
  private measurementLabel: HTMLDivElement;
  private measurementStartLabel: HTMLDivElement;
  private measurementEndLabel: HTMLDivElement;
  // Active measurement session, if any (see `beginMeasurementSession`).
  private measureSession: { cleanup: () => void } | undefined;

  private colorFactor = vec4.fromValues(1, 1, 1, 1);
  private pickIDs = new PickIDManager();

  flushBackendProjectionParameters() {
    this.sliceView.flushBackendProjectionParameters();
  }

  private visibleLayerTracker: VisibleRenderLayerTracker<
    SliceViewPanel,
    SliceViewPanelRenderLayer
  >;

  get displayDimensionRenderInfo() {
    return this.navigationState.displayDimensionRenderInfo;
  }

  // FIXME: use separate backend object for the panel
  get rpc() {
    return this.sliceView.rpc!;
  }
  get rpcId() {
    return this.sliceView.rpcId!;
  }

  private offscreenFramebuffer = this.registerDisposer(
    new FramebufferConfiguration(this.gl, {
      colorBuffers: [
        new TextureBuffer(
          this.gl,
          WebGL2RenderingContext.RGBA8,
          WebGL2RenderingContext.RGBA,
          WebGL2RenderingContext.UNSIGNED_BYTE,
        ),
        new TextureBuffer(
          this.gl,
          WebGL2RenderingContext.R32F,
          WebGL2RenderingContext.RED,
          WebGL2RenderingContext.FLOAT,
        ),
      ],
    }),
  );

  private offscreenCopyHelper = this.registerDisposer(
    OffscreenCopyHelper.get(this.gl),
  );
  private scaleBars = this.registerDisposer(
    new MultipleScaleBarTextures(this.gl),
  );

  get navigationState() {
    return this.sliceView.navigationState;
  }

  constructor(
    context: Borrowed<DisplayContext>,
    element: HTMLElement,
    public sliceView: Owned<SliceView>,
    viewer: SliceViewerState,
  ) {
    super(context, element, viewer);

    this.sliceViewRenderHelper = this.registerDisposer(
      SliceViewRenderHelper.get(
        this.gl,
        sliceViewPanelEmitColor,
        this.viewer,
        false /*sliceViewPanel*/,
      ),
    );

    viewer.wireFrame.changed.add(() => this.scheduleRedraw());

    const makeMeasurementLabel = () => {
      const label = document.createElement("div");
      label.className = "neuroglancer-measurement-label";
      Object.assign(label.style, {
        position: "absolute",
        display: "none",
        pointerEvents: "none",
        transform: "translate(-50%, -50%)",
        padding: "1px 4px",
        borderRadius: "3px",
        background: "rgba(0,0,0,0.6)",
        color: "#00ffff",
        font: "10px sans-serif",
        whiteSpace: "nowrap",
        textAlign: "center",
        zIndex: "10",
      });
      element.appendChild(label);
      this.registerDisposer(() => removeFromParent(label));
      return label;
    };
    this.measurementLabel = makeMeasurementLabel();
    // The length label sits above the midpoint; endpoint labels are centered on
    // an offset point just beyond each end (see updateMeasurementLabel).
    this.measurementLabel.style.transform = "translate(-50%, -150%)";
    this.measurementStartLabel = makeMeasurementLabel();
    this.measurementEndLabel = makeMeasurementLabel();
    this.registerDisposer(() => this.measureSession?.cleanup());
    this.registerDisposer(
      viewer.measurementState.changed.add(() => {
        if (this.visible) this.scheduleRedraw();
      }),
    );
    this.registerDisposer(
      viewer.coordinateDisplayMode.changed.add(() => {
        if (this.visible) this.scheduleRedraw();
      }),
    );
    registerActionListener(
      element,
      "measure-line",
      (e: ActionEvent<MouseEvent>) =>
        this.beginMeasurementSession(e, "line", 0 /* left button */),
    );
    registerActionListener(
      element,
      "measure-box",
      (e: ActionEvent<MouseEvent>) =>
        this.beginMeasurementSession(e, "box", 2 /* right button */),
    );
    registerActionListener(element, "clear-measurement", () => {
      this.measureSession?.cleanup();
      this.viewer.measurementState.clear();
    });

    registerActionListener(
      element,
      "rotate-via-mouse-drag",
      (e: ActionEvent<MouseEvent>) => {
        const { mouseState } = this.viewer;
        if (mouseState.updateUnconditionally()) {
          const initialPosition = Float32Array.from(mouseState.position);
          startRelativeMouseDrag(e.detail, (_event, deltaX, deltaY) => {
            this.context.flagContinuousCameraMotion();
            const { pose } = this.navigationState;
            const xAxis = vec3.transformQuat(
              tempVec3,
              kAxes[0],
              pose.orientation.orientation,
            );
            const yAxis = vec3.transformQuat(
              tempVec3b,
              kAxes[1],
              pose.orientation.orientation,
            );
            this.viewer.navigationState.pose.rotateAbsolute(
              yAxis,
              ((-deltaX / 4.0) * Math.PI) / 180.0,
              initialPosition,
            );
            this.viewer.navigationState.pose.rotateAbsolute(
              xAxis,
              ((-deltaY / 4.0) * Math.PI) / 180.0,
              initialPosition,
            );
          });
        }
      },
    );

    registerActionListener(
      element,
      "rotate-in-plane-via-touchrotate",
      (e: ActionEvent<TouchRotateInfo>) => {
        const { detail } = e;
        const { mouseState } = this.viewer;
        this.handleMouseMove(detail.centerX, detail.centerY);
        if (mouseState.updateUnconditionally()) {
          this.context.flagContinuousCameraMotion();
          this.navigationState.pose.rotateAbsolute(
            this.sliceView.projectionParameters.value
              .viewportNormalInCanonicalCoordinates,
            detail.angle - detail.prevAngle,
            mouseState.position,
          );
        }
      },
    );

    this.registerDisposer(sliceView);
    // Create visible layer tracker after registering SliceView, to ensure it is destroyed before
    // SliceView backend is destroyed.
    this.visibleLayerTracker = makeRenderedPanelVisibleLayerTracker(
      this.viewer.layerManager,
      SliceViewPanelRenderLayer,
      this.viewer.visibleLayerRoles,
      this,
    );

    this.registerDisposer(
      viewer.crossSectionBackgroundColor.changed.add(() =>
        this.scheduleRedraw(),
      ),
    );
    this.registerDisposer(sliceView.visibility.add(this.visibility));
    this.registerDisposer(
      sliceView.viewChanged.add(() => {
        if (this.visible) {
          context.scheduleRedraw();
        }
      }),
    );
    this.registerDisposer(
      viewer.showAxisLines.changed.add(() => {
        if (this.visible) {
          this.scheduleRedraw();
        }
      }),
    );
    this.registerDisposer(
      viewer.showCrossSectionHoverPosition.changed.add(() => {
        if (this.visible) {
          this.scheduleRedraw();
        }
      }),
    );
    this.registerDisposer(
      viewer.mouseState.changed.add(() => {
        if (this.visible && viewer.showCrossSectionHoverPosition.value) {
          this.scheduleRedraw();
        }
      }),
    );

    this.registerDisposer(
      viewer.showScaleBar.changed.add(() => {
        if (this.visible) {
          this.context.scheduleRedraw();
        }
      }),
    );
    this.registerDisposer(
      viewer.scaleBarOptions.changed.add(() => {
        if (this.visible) {
          this.context.scheduleRedraw();
        }
      }),
    );
  }

  translateByViewportPixels(deltaX: number, deltaY: number): void {
    const { pose } = this.viewer.navigationState;
    pose.updateDisplayPosition((pos) => {
      vec3.set(pos, -deltaX, -deltaY, 0);
      vec3.transformMat4(
        pos,
        pos,
        this.sliceView.projectionParameters.value.invViewMatrix,
      );
    });
  }

  translateDataPointByViewportPixels(
    out: vec3,
    orig: vec3,
    deltaX: number,
    deltaY: number,
  ): vec3 {
    const projectionParameters = this.sliceView.projectionParameters.value;
    vec3.transformMat4(out, orig, projectionParameters.viewMatrix);
    vec3.set(out, out[0] + deltaX, out[1] + deltaY, out[2]);
    vec3.transformMat4(out, out, projectionParameters.invViewMatrix);
    return out;
  }

  isReady() {
    if (!this.visible) {
      return false;
    }

    const { sliceView } = this;

    this.ensureBoundsUpdated();

    if (!sliceView.isReady()) {
      return false;
    }

    const renderContext: SliceViewPanelReadyRenderContext = {
      projectionParameters: sliceView.projectionParameters.value,
      sliceView,
    };

    for (const [renderLayer, attachment] of this.visibleLayerTracker
      .visibleLayers) {
      if (!renderLayer.isReady(renderContext, attachment)) {
        return false;
      }
    }
    return true;
  }

  drawWithPicking(pickingData: FramePickingData): boolean {
    const { sliceView } = this;
    if (!sliceView.valid) {
      return false;
    }
    sliceView.updateRendering();
    const projectionParameters = sliceView.projectionParameters.value;
    const { width, height, invViewProjectionMat } = projectionParameters;
    mat4.copy(pickingData.invTransform, invViewProjectionMat);
    const { gl } = this;

    this.offscreenFramebuffer.bind(width, height);
    gl.disable(WebGL2RenderingContext.SCISSOR_TEST);
    this.gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);

    const backgroundColor = tempVec4;
    const crossSectionBackgroundColor =
      this.viewer.crossSectionBackgroundColor.value;
    backgroundColor[0] = crossSectionBackgroundColor[0];
    backgroundColor[1] = crossSectionBackgroundColor[1];
    backgroundColor[2] = crossSectionBackgroundColor[2];
    backgroundColor[3] = 1;

    this.offscreenFramebuffer.bindSingle(OffscreenTextures.COLOR);
    this.sliceViewRenderHelper.draw(
      sliceView.offscreenFramebuffer.colorBuffers[0].texture,
      identityMat4,
      this.colorFactor,
      backgroundColor,
      0,
      0,
      1,
      1,
    );

    const { visibleLayers } = this.visibleLayerTracker;
    const { pickIDs } = this;
    pickIDs.clear();

    const bindFramebuffer = () => {
      gl.disable(WebGL2RenderingContext.SCISSOR_TEST);
      gl.enable(WebGL2RenderingContext.BLEND);
      gl.blendFunc(
        WebGL2RenderingContext.SRC_ALPHA,
        WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA,
      );
      this.offscreenFramebuffer.bind(width, height);
    };

    bindFramebuffer();

    const renderContext: SliceViewPanelRenderContext = {
      wireFrame: this.viewer.wireFrame.value,
      projectionParameters,
      pickIDs: pickIDs,
      emitter: sliceViewPanelEmitColorAndPickID,
      emitColor: true,
      emitPickID: true,
      sliceView,
      bindFramebuffer,
      frameNumber: this.context.frameNumber,
    };
    for (const [renderLayer, attachment] of visibleLayers) {
      renderLayer.draw(renderContext, attachment);
    }
    gl.disable(WebGL2RenderingContext.BLEND);
    const { mouseState } = this.viewer;
    // Draw the hover position coming from another panel, but not in the panel
    // the mouse is currently over (`mouseX >= 0`), where the real cursor is
    // already visible.
    const showHoverMarker = shouldDrawCrossSectionHoverMarker({
      enabled: this.viewer.showCrossSectionHoverPosition.value,
      mouseActive: mouseState.active,
      mouseInThisPanel: this.mouseX >= 0,
    });
    const showMeasurement = this.showMeasurementInThisPanel();
    if (
      this.viewer.showAxisLines.value ||
      this.viewer.showScaleBar.value ||
      showHoverMarker ||
      showMeasurement
    ) {
      this.offscreenFramebuffer.bindSingle(OffscreenTextures.COLOR);
      if (this.viewer.showAxisLines.value) {
        const axisLength =
          (Math.min(
            projectionParameters.logicalWidth,
            projectionParameters.logicalHeight,
          ) /
            4) *
          1.5;
        const {
          zoomFactor: { value: zoom },
        } = this.viewer.navigationState;
        this.axesLineHelper.draw(
          disableZProjection(
            computeAxisLineMatrix(projectionParameters, axisLength * zoom),
          ),
        );
      }
      if (showHoverMarker) {
        // A fixed-size, screen-aligned crosshair so it reads consistently in
        // every orthoview regardless of the panel's orientation.
        const markerMat = computeHoverMarkerMatrix(
          projectionParameters,
          HOVER_MARKER_SIZE,
          mouseState.position,
        );
        // Fade the marker with distance from this panel's slice plane.
        const alpha = crossSectionHoverMarkerAlpha(markerMat);
        if (alpha > 0) {
          vec4.set(tempHoverColor, 1, 0.85, 0, alpha);
          this.hoverMarker.draw(disableZProjection(markerMat), tempHoverColor);
        }
      }
      if (showMeasurement) {
        const { measurementState } = this.viewer;
        const a = this.projectToNdc(
          projectionParameters,
          measurementState.startPosition!,
        );
        const b = this.projectToNdc(
          projectionParameters,
          measurementState.endPosition!,
        );
        const draw = (p: NdcPoint, q: NdcPoint) =>
          this.measurementLineHelper.draw(p, q, this.measurementColor);
        if (measurementState.shape === "box") {
          // The rectangle is axis-aligned on screen, so its two extra corners
          // are formed by crossing the projected endpoints' x and y.  Deriving
          // them after projection (rather than by swapping one global display
          // dimension of each endpoint) is what makes the box correct on any
          // cross section: `displayDimensionIndices` is global, identical for
          // the xy/xz/yz panels, so display dims 0 and 1 are only both in-plane
          // in the xy panel.  Elsewhere one of them is the plane normal, which
          // pushed two corners out of the slice and collapsed the rectangle
          // into a pair of edges plus a diagonal.
          const c: NdcPoint = [b[0], a[1]];
          const d: NdcPoint = [a[0], b[1]];
          draw(a, c);
          draw(c, b);
          draw(b, d);
          draw(d, a);
        } else {
          draw(a, b);
        }
      }
      if (this.viewer.showScaleBar.value) {
        gl.enable(WebGL2RenderingContext.BLEND);
        gl.blendFunc(
          WebGL2RenderingContext.SRC_ALPHA,
          WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA,
        );
        this.scaleBars.draw(
          projectionParameters,
          this.navigationState.displayDimensionRenderInfo.value,
          this.navigationState.relativeDisplayScales.value,
          this.navigationState.zoomFactor.value,
          this.viewer.scaleBarOptions.value,
        );
        gl.disable(WebGL2RenderingContext.BLEND);
      }
    }

    this.offscreenFramebuffer.unbind();

    this.updateMeasurementLabel(projectionParameters, showMeasurement);

    // Draw the texture over the whole viewport.
    this.setGLClippedViewport();
    this.offscreenCopyHelper.draw(
      this.offscreenFramebuffer.colorBuffers[OffscreenTextures.COLOR].texture,
    );
    return true;
  }

  // Starts a measurement *session* for the given shape.  Both shapes share the
  // shift+alt chord and differ only by button: left draws a line, right draws a
  // box.  The session stays alive as long as the alt key is held: pressing the
  // trigger button positions the end point/opposite corner, releasing the button
  // merely pauses (the end freezes), and pressing again resumes -- convenient on
  // a trackpad where holding a press through a drag is awkward.  The measurement
  // is finalized only once alt is released *and* the button is up.  (Shift is
  // only needed to start; releasing it mid-session does not end the session.)  A
  // press that reaches this handler while a session is already active is handled
  // by the session's own listeners, so it is ignored here.
  private beginMeasurementSession(
    e: ActionEvent<MouseEvent>,
    shape: MeasurementShape,
    button: number,
  ) {
    if (this.measureSession !== undefined) return;
    const { element } = this;
    const { mouseState, measurementState } = this.viewer;
    this.handleMouseMove(e.detail.clientX, e.detail.clientY);
    if (!mouseState.updateUnconditionally()) return;
    measurementState.begin(
      shape,
      mouseState.position,
      this.navigationState.pose.orientation.orientation,
      this.navigationState.coordinateSpace.value,
    );
    const { document: doc } = e.detail.view!;
    const session = { buttonDown: true, altHeld: true };
    // Reused scratch for the clamped endpoint (`update` copies it).
    let clampedEnd = new Float32Array(mouseState.position.length);
    // Updates the end point to follow the cursor, clamped to the volume bounding
    // box.  Returns whether the cursor was inside the volume; when it is outside,
    // the end point sticks to the boundary and this returns false so callers can
    // stop auto-panning past the edge of the data.
    const updateEnd = (): boolean => {
      if (!mouseState.updateUnconditionally()) return false;
      const { position } = mouseState;
      if (clampedEnd.length !== position.length) {
        clampedEnd = new Float32Array(position.length);
      }
      const inBounds = clampPositionToBounds(
        clampedEnd,
        position,
        this.navigationState.coordinateSpace.value,
      );
      measurementState.update(clampedEnd);
      return inBounds;
    };
    // While the button is held and the cursor sits on a panel edge, scroll the
    // view to reveal content past that edge so the measurement can be extended
    // beyond the visible area.  Panning is driven by pointermove events, so it
    // only happens while the user is actively moving the mouse -- resting the
    // cursor at the edge does not keep scrolling the view.
    const doEdgePan = () => {
      if (!session.buttonDown) return;
      const [vx, vy] = computeEdgePanVelocity(
        this.mouseX,
        this.mouseY,
        element.offsetWidth,
        element.offsetHeight,
      );
      if (vx === 0 && vy === 0) return;
      this.context.flagContinuousCameraMotion();
      this.translateByViewportPixels(vx, vy);
      this.scheduleRedraw();
    };
    const onMove = (event: PointerEvent) => {
      session.altHeld = event.altKey;
      this.handleMouseMove(event.clientX, event.clientY);
      if (session.buttonDown) {
        // Only auto-pan while the cursor is still inside the volume; once it
        // leaves, the end point clamps to the boundary and the view holds.
        if (updateEnd()) doEdgePan();
      }
      maybeFinish();
    };
    const onDown = (event: PointerEvent) => {
      if (event.button !== button) return;
      session.buttonDown = true;
      session.altHeld = event.altKey;
      this.handleMouseMove(event.clientX, event.clientY);
      updateEnd();
    };
    const onUp = (event: PointerEvent) => {
      if (event.button !== button) return;
      session.buttonDown = false;
      session.altHeld = event.altKey;
      maybeFinish();
    };
    const onKey = (event: KeyboardEvent) => {
      session.altHeld = event.altKey;
      maybeFinish();
    };
    const preventContextMenu = (event: Event) => event.preventDefault();
    const cleanup = () => {
      doc.removeEventListener("pointermove", onMove, true);
      doc.removeEventListener("pointerdown", onDown, true);
      doc.removeEventListener("pointerup", onUp, true);
      doc.removeEventListener("keydown", onKey, true);
      doc.removeEventListener("keyup", onKey, true);
      doc.removeEventListener("contextmenu", preventContextMenu, true);
      this.measureSession = undefined;
    };
    const maybeFinish = () => {
      // Terminate only when neither the alt key nor the trigger button is
      // engaged.
      if (!session.buttonDown && !session.altHeld) {
        cleanup();
        measurementState.finish();
      }
    };
    this.measureSession = { cleanup };
    doc.addEventListener("pointermove", onMove, true);
    doc.addEventListener("pointerdown", onDown, true);
    doc.addEventListener("pointerup", onUp, true);
    doc.addEventListener("keydown", onKey, true);
    doc.addEventListener("keyup", onKey, true);
    doc.addEventListener("contextmenu", preventContextMenu, true);
  }

  // A measurement is shown only in panels displaying (nearly) the plane it was
  // drawn in, compared by plane normal rather than by orientation equality.
  // Comparing the full quaternion made any rotation at all -- including a purely
  // in-plane one, which does not change the plane -- silently hide the
  // measurement; and with an exact comparison even a fraction of a degree of
  // accidental tilt did, which is easy to trigger since rotate-via-mouse-drag
  // shares the measurement's modifier chord minus alt.  The dot product is taken
  // as an absolute value so a plane viewed from the opposite side still matches.
  private showMeasurementInThisPanel(): boolean {
    const { measurementState } = this.viewer;
    if (!measurementState.isSet) return false;
    const panelNormal = vec3.transformQuat(
      tempVec3c,
      kAxes[2],
      this.navigationState.pose.orientation.orientation,
    );
    const measurementNormal = vec3.transformQuat(
      tempVec3d,
      kAxes[2],
      measurementState.orientation,
    );
    return (
      Math.abs(vec3.dot(panelNormal, measurementNormal)) >=
      MEASURE_PLANE_MATCH_MIN_COS
    );
  }

  // Projects a global-coordinate point to normalized device coordinates using
  // this frame's view-projection, mapping global dimensions to display
  // dimensions exactly as `computeAxisLineMatrix` does.  Both the drawn geometry
  // and the HTML labels are placed from this, so they cannot disagree.
  private projectToNdc(
    projectionParameters: ProjectionParameters,
    position: Float32Array,
  ): NdcPoint {
    const { displayDimensionIndices } =
      projectionParameters.displayDimensionRenderInfo;
    const p = tempVec4b;
    for (let i = 0; i < 3; ++i) {
      const d = displayDimensionIndices[i];
      p[i] = d === -1 || d >= position.length ? 0 : position[d];
    }
    p[3] = 1;
    vec4.transformMat4(p, p, projectionParameters.viewProjectionMat);
    const w = p[3] || 1;
    return [p[0] / w, p[1] / w];
  }

  // Converts a point in normalized device coordinates to the panel's logical
  // pixel space, for positioning the HTML measurement labels.
  private ndcToPixels(
    projectionParameters: ProjectionParameters,
    p: NdcPoint,
  ): [number, number] {
    return [
      (p[0] * 0.5 + 0.5) * projectionParameters.logicalWidth,
      (1 - (p[1] * 0.5 + 0.5)) * projectionParameters.logicalHeight,
    ];
  }

  private updateMeasurementLabel(
    projectionParameters: ProjectionParameters,
    show: boolean,
  ) {
    const label = this.measurementLabel;
    const startLabel = this.measurementStartLabel;
    const endLabel = this.measurementEndLabel;
    if (!show) {
      label.style.display = "none";
      startLabel.style.display = "none";
      endLabel.style.display = "none";
      return;
    }
    const { measurementState } = this.viewer;
    const start = measurementState.startPosition!;
    const end = measurementState.endPosition!;
    const coordinateSpace =
      measurementState.coordinateSpace ??
      this.navigationState.coordinateSpace.value;
    const mode = this.viewer.coordinateDisplayMode.value;
    const [sx, sy] = this.ndcToPixels(
      projectionParameters,
      this.projectToNdc(projectionParameters, start),
    );
    const [ex, ey] = this.ndcToPixels(
      projectionParameters,
      this.projectToNdc(projectionParameters, end),
    );
    const mx = (sx + ex) / 2;
    const my = (sy + ey) / 2;
    // A box shows only its two corner coordinates; a line also shows its length
    // at the midpoint.
    if (measurementState.shape === "box") {
      label.style.display = "none";
    } else {
      label.textContent = formatMeasurementLength(
        start,
        end,
        coordinateSpace,
        mode,
      );
      label.style.left = `${mx}px`;
      label.style.top = `${my}px`;
      label.style.display = "";
    }
    // Place each endpoint's coordinate label just outside its end, offset along
    // the line direction (away from the midpoint) so it does not cover the line.
    const OFFSET = 16;
    const placeEndpoint = (
      endpointLabel: HTMLDivElement,
      text: string,
      px: number,
      py: number,
    ) => {
      endpointLabel.textContent = text;
      let dx = px - mx;
      let dy = py - my;
      const len = Math.hypot(dx, dy);
      if (len < 1e-3) {
        dx = 0;
        dy = -1;
      } else {
        dx /= len;
        dy /= len;
      }
      endpointLabel.style.left = `${px + dx * OFFSET}px`;
      endpointLabel.style.top = `${py + dy * OFFSET}px`;
      endpointLabel.style.display = "";
    };
    placeEndpoint(
      startLabel,
      formatPosition(start, coordinateSpace, mode),
      sx,
      sy,
    );
    placeEndpoint(endLabel, formatPosition(end, coordinateSpace, mode), ex, ey);
  }

  ensureBoundsUpdated() {
    super.ensureBoundsUpdated(true /* canScaleForScreenshot */);
    this.sliceView.projectionParameters.setViewport(this.renderViewport);
  }

  issuePickRequest(glWindowX: number, glWindowY: number, pickRadius: number) {
    const { offscreenFramebuffer } = this;
    const pickDiameter = getPickDiameter(pickRadius);
    offscreenFramebuffer.readPixelFloat32IntoBuffer(
      OffscreenTextures.PICK,
      glWindowX - pickRadius,
      glWindowY - pickRadius,
      0,
      pickDiameter,
      pickDiameter,
    );
  }

  completePickRequest(
    glWindowX: number,
    glWindowY: number,
    data: Float32Array,
    pickingData: FramePickingData,
    pickRadius: number,
  ) {
    const { mouseState } = this.viewer;
    mouseState.pickedRenderLayer = null;
    const pickOffsetSequence = getPickOffsetSequence(pickRadius);
    const { viewportWidth, viewportHeight } = pickingData;
    const { value: voxelCoordinates } = this.navigationState.position;
    const rank = voxelCoordinates.length;
    const displayDimensions = this.navigationState.pose.displayDimensions.value;
    const { displayRank, displayDimensionIndices } = displayDimensions;

    const setPosition = (
      relativeX: number,
      relativeY: number,
      position: Float32Array,
    ) => {
      tempVec3[0] =
        (2.0 * (glWindowX + relativeX - pickRadius)) / viewportWidth - 1.0;
      tempVec3[1] =
        (2.0 * (glWindowY + relativeY - pickRadius)) / viewportHeight - 1.0;
      tempVec3[2] = 0;
      vec3.transformMat4(tempVec3, tempVec3, pickingData.invTransform);
      position.set(voxelCoordinates);
      for (let i = 0; i < displayRank; ++i) {
        position[displayDimensionIndices[i]] = tempVec3[i];
      }
    };

    let { unsnappedPosition } = mouseState;
    if (unsnappedPosition.length !== rank) {
      unsnappedPosition = mouseState.unsnappedPosition = new Float32Array(rank);
    }
    mouseState.coordinateSpace = this.navigationState.coordinateSpace.value;
    mouseState.displayDimensions = displayDimensions;

    setPosition(pickRadius, pickRadius, unsnappedPosition);

    const setStateFromRelative = (
      relativeX: number,
      relativeY: number,
      pickId: number,
    ) => {
      let { position: mousePosition } = mouseState;
      if (mousePosition.length !== rank) {
        mousePosition = mouseState.position = new Float32Array(rank);
      }
      setPosition(relativeX, relativeY, mousePosition);
      this.pickIDs.setMouseState(mouseState, pickId);
      mouseState.setActive(true);
    };
    const resolvedPick = resolveNearestPanelPickSample(
      data,
      pickOffsetSequence,
      pickRadius,
    );
    if (resolvedPick !== undefined) {
      setStateFromRelative(
        resolvedPick.relativeX,
        resolvedPick.relativeY,
        resolvedPick.pickValue,
      );
      return;
    }
    setStateFromRelative(pickRadius, pickRadius, 0);
  }

  /**
   * Zooms by the specified factor, maintaining the data position that projects to the current mouse
   * position.
   */
  zoomByMouse(factor: number) {
    const { navigationState } = this;
    if (!navigationState.valid) {
      return;
    }
    const { sliceView } = this;
    const {
      width,
      height,
      invViewMatrix,
      displayDimensionRenderInfo: { displayDimensionIndices, displayRank },
    } = sliceView.projectionParameters.value;
    let { mouseX, mouseY } = this;
    mouseX -= width / 2;
    mouseY -= height / 2;
    // Desired invariance:
    //
    // invViewMatrixLinear * [mouseX, mouseY, 0]^T + [oldX, oldY, oldZ]^T =
    // invViewMatrixLinear * factor * [mouseX, mouseY, 0]^T + [newX, newY, newZ]^T

    const position = this.navigationState.position.value;
    for (let i = 0; i < displayRank; ++i) {
      const dim = displayDimensionIndices[i];
      const f = invViewMatrix[i] * mouseX + invViewMatrix[4 + i] * mouseY;
      position[dim] += f * (1 - factor);
    }
    this.navigationState.position.changed.dispatch();
    navigationState.zoomBy(factor);
  }
}
