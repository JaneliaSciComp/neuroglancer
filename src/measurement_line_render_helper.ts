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
 * @file Draws the measurement ruler segment in a cross-section slice panel.
 */

import { RefCounted } from "#src/util/disposable.js";
import type { vec4 } from "#src/util/geom.js";
import { identityMat4 } from "#src/util/geom.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import type { GL } from "#src/webgl/context.js";
import type { ShaderProgram } from "#src/webgl/shader.js";
import { trivialUniformColorShader } from "#src/webgl/trivial_shaders.js";

/**
 * A point of measurement geometry in normalized device coordinates (x and y in
 * [-1, 1], y pointing up), as produced by `SliceViewPanel.projectToNdc`.
 */
export type NdcPoint = readonly [number, number];

export class MeasurementLineRenderHelper extends RefCounted {
  private vertexBuffer: GLBuffer;
  private data = new Float32Array(8); // two vec4 endpoints
  private shader: ShaderProgram;

  constructor(public gl: GL) {
    super();
    this.vertexBuffer = this.registerDisposer(new GLBuffer(gl));
    this.shader = trivialUniformColorShader(gl);
  }

  static get(gl: GL) {
    return gl.memoize.get(
      "SliceViewPanel:MeasurementLineRenderHelper",
      () => new MeasurementLineRenderHelper(gl),
    );
  }

  /**
   * Draws a segment between two points already projected to normalized device
   * coordinates, with an identity projection.
   *
   * Projecting on the CPU rather than passing global coordinates through
   * `viewProjectionMat` in the shader keeps the clip-space depth pinned to 0, so
   * measurement geometry is never depth-clipped out of a thin cross section --
   * the same reason `AxesLineHelper` is fed a `disableZProjection`ed matrix.
   * Without this, the sub-voxel out-of-plane component that a picked position
   * carries (`mouseState.position` is a Float32Array) is enough to make a
   * segment vanish at some zoom levels.
   */
  draw(a: NdcPoint, b: NdcPoint, color: vec4) {
    const { shader, gl, data } = this;
    data[0] = a[0];
    data[1] = a[1];
    data[2] = 0;
    data[3] = 1;
    data[4] = b[0];
    data[5] = b[1];
    data[6] = 0;
    data[7] = 1;
    this.vertexBuffer.setData(data, gl.DYNAMIC_DRAW);

    shader.bind();
    gl.uniformMatrix4fv(
      shader.uniform("uProjectionMatrix"),
      false,
      identityMat4,
    );
    gl.uniform4fv(shader.uniform("uColor"), color);
    const aVertexPosition = shader.attribute("aVertexPosition");
    this.vertexBuffer.bindToVertexAttrib(aVertexPosition, 4);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.lineWidth(1);
    gl.drawArrays(gl.LINES, 0, 2);
    gl.disable(gl.BLEND);

    gl.disableVertexAttribArray(aVertexPosition);
  }
}
