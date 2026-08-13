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
 * Refines a clicked position to the fitted center of the surrounding image intensity, so that a
 * placed or dragged annotation vertex lands on a feature rather than wherever the cursor happened
 * to be.
 *
 * This module deliberately does not import `#src/ui/annotations.js`, which imports it.
 */

import type { AnnotationLayerState } from "#src/annotation/annotation_layer_state.js";
import type { Annotation } from "#src/annotation/index.js";
import { AnnotationType } from "#src/annotation/index.js";
import type { PointFitter, VoxelPatch } from "#src/annotation/point_fit.js";
import { getPointFitter, normalizePatch } from "#src/annotation/point_fit.js";
import type { MouseSelectionState, UserLayer } from "#src/layer/index.js";
import { getChunkPositionFromCombinedGlobalLocalPositions } from "#src/render_coordinate_transform.js";
import { ImageRenderLayer } from "#src/sliceview/volume/image_renderlayer.js";
import { StatusMessage } from "#src/status.js";

/**
 * Whether `ann` has a vertex that can meaningfully be moved to a fitted feature center. A
 * bounding box corner or an ellipsoid radius is an extent, not a feature center, so fitting one
 * would corrupt the geometry.
 */
export function canFitAnnotation(ann: Annotation): boolean {
  switch (ann.type) {
    case AnnotationType.POINT:
    case AnnotationType.LINE:
    case AnnotationType.POLYLINE:
      return true;
    default:
      return false;
  }
}

/**
 * Converts `globalPosition`, in the global coordinate space, to the coordinate space in which
 * annotations of `annotationLayer` are expressed.  Returns `undefined` if the position lies
 * outside the annotation layer's clip bounds.
 */
export function getGlobalPositionInAnnotationCoordinates(
  globalPosition: Float32Array,
  annotationLayer: AnnotationLayerState,
): Float32Array | undefined {
  const chunkTransform = annotationLayer.chunkTransform.value;
  if (chunkTransform.error !== undefined) return undefined;
  const chunkPosition = new Float32Array(
    chunkTransform.modelTransform.unpaddedRank,
  );
  if (
    !getChunkPositionFromCombinedGlobalLocalPositions(
      chunkPosition,
      globalPosition,
      annotationLayer.localPosition.value,
      chunkTransform.layerRank,
      chunkTransform.combinedGlobalLocalToChunkTransform,
    )
  ) {
    return undefined;
  }
  return chunkPosition;
}

export const DEFAULT_FIT_RADIUS = 5;
export const MAX_FIT_RADIUS = 32;

export interface FitSettings {
  /** Half-width of the sampled cube, in voxels. */
  radius: number;
  /** Name of a fitter registered in `#src/annotation/point_fit.js`. */
  method: string;
  /** Fit a dark blob on a bright background, rather than a bright blob on a dark background. */
  invert: boolean;
  /** Fraction of the peak height below which samples are excluded from the fit. */
  relativeThreshold: number;
  /** Minimum number of above-threshold samples required before attempting a fit. */
  minSamples: number;
}

/**
 * Samples a cube of image intensity centered on `center`, aligned to the three display dimensions
 * with a spacing of one unit of the global coordinate space.
 *
 * Sampling in global coordinates, rather than in the image's chunk coordinates, means the fitted
 * center is produced directly in the global space and needs no inverse transform to get back out.
 *
 * Samples are taken at voxel *centers*.  `getValueAt` floors the position to get a voxel index, so
 * voxel `i` spans the continuous interval `[i, i+1)` and is centered at `i + 0.5`.  The returned
 * `origin` is therefore the half-integer coordinate of sample `[0,0,0]`, which lets the caller map
 * a fractional patch index straight back to a continuous coordinate by adding the two.  Getting
 * this wrong biases every fitted point by half a voxel on each axis.
 *
 * Returns `undefined` if fewer than half the samples could be read, which is the case when the
 * region has not been loaded.
 */
export function samplePatch(
  renderLayer: ImageRenderLayer,
  center: Float32Array,
  displayDimensionIndices: Int32Array,
  radius: number,
): { patch: VoxelPatch; origin: [number, number, number] } | undefined {
  const origin: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; ++k) {
    const globalDim = displayDimensionIndices[k];
    // A rank < 3 display space cannot support a 3-d fit.
    if (globalDim === -1) return undefined;
    origin[k] = Math.floor(center[globalDim]) - radius + 0.5;
  }
  const n = 2 * radius + 1;
  const position = Float32Array.from(center);
  const data = new Float32Array(n * n * n);
  let missing = 0;
  let i = 0;
  // ponytail: one getValueAt call per sample, each a chunk-grid Map lookup keyed by a joined
  // string.  1331 lookups at the default radius, comfortably under a frame; resolve the chunk
  // once and index into it directly if the radius ever needs to grow much past this.
  for (let z = 0; z < n; ++z) {
    position[displayDimensionIndices[2]] = origin[2] + z;
    for (let y = 0; y < n; ++y) {
      position[displayDimensionIndices[1]] = origin[1] + y;
      for (let x = 0; x < n; ++x, ++i) {
        position[displayDimensionIndices[0]] = origin[0] + x;
        // `null` when no visible source has the chunk resident, `undefined` when the position
        // falls outside the chunk's data bounds.
        const value = renderLayer.getValueAt(position);
        if (value == null) {
          data[i] = Number.NaN;
          ++missing;
          continue;
        }
        // ponytail: multi-channel images are fit on channel 0.
        data[i] = Number(Array.isArray(value) ? value[0] : value);
      }
    }
  }
  if (missing * 2 > data.length) return undefined;
  return { patch: { data, size: [n, n, n] }, origin };
}

/**
 * Picks the image data to fit against: the annotation layer itself when annotations live directly
 * on an image layer, otherwise the first visible image layer.
 *
 * ponytail: no way to disambiguate between several visible image layers.  `LayerReference` plus
 * `LayerReferenceWidget` is the drop-in precedent if that becomes necessary.
 */
function findImageRenderLayer(
  layer: UserLayer,
): { renderLayer: ImageRenderLayer; name: string } | undefined {
  const candidates: UserLayer[] = [layer];
  for (const managed of layer.manager.rootLayers.managedLayers) {
    if (managed.visible && managed.layer != null) {
      candidates.push(managed.layer);
    }
  }
  for (const candidate of candidates) {
    for (const renderLayer of candidate.renderLayers) {
      if (renderLayer instanceof ImageRenderLayer) {
        return { renderLayer, name: candidate.managedLayer.name };
      }
    }
  }
  return undefined;
}

/**
 * Returns the fitted position, in the global coordinate space, of the feature under the mouse.
 *
 * Falls back to the raw cursor position, with an explanatory status message, whenever the fit
 * cannot be performed — a click is never silently discarded, and never lands somewhere the user
 * did not indicate.
 */
export function fitGlobalPosition(
  layer: UserLayer,
  mouseState: MouseSelectionState,
  settings: FitSettings,
): Float32Array {
  const clicked = Float32Array.from(mouseState.unsnappedPosition);
  const source = findImageRenderLayer(layer);
  if (source === undefined) {
    StatusMessage.showTemporaryMessage(
      "Cannot snap to fit: no visible image layer to read. " +
        "Placed at the clicked position.",
    );
    return clicked;
  }
  const { displayDimensions } = mouseState;
  if (displayDimensions === undefined) {
    StatusMessage.showTemporaryMessage(
      "Cannot snap to fit: no display dimensions for the cursor position. " +
        "Placed at the clicked position.",
    );
    return clicked;
  }
  const sampled = samplePatch(
    source.renderLayer,
    clicked,
    displayDimensions.displayDimensionIndices,
    settings.radius,
  );
  if (sampled === undefined) {
    StatusMessage.showTemporaryMessage(
      `Cannot snap to fit: image data around the cursor in "${source.name}" is not loaded. ` +
        "Placed at the clicked position.",
    );
    return clicked;
  }
  // Fitters assume a fixed, bounded intensity range; the source image may be arbitrary floats.
  // They also fit a bright peak, so a dark feature must be inverted first.
  normalizePatch(sampled.patch, settings.invert);
  const fit: PointFitter | undefined = getPointFitter(settings.method);
  const center = fit?.(sampled.patch, {
    relativeThreshold: settings.relativeThreshold,
    minSamples: settings.minSamples,
  });
  if (center === undefined) {
    StatusMessage.showTemporaryMessage(
      `Cannot snap to fit: the ${settings.method} fit did not converge within ` +
        `${settings.radius} voxels of the cursor. Placed at the clicked position. ` +
        "Click closer to the feature, or increase the fit radius.",
    );
    return clicked;
  }
  const fitted = Float32Array.from(clicked);
  const { displayDimensionIndices } = displayDimensions;
  for (let k = 0; k < 3; ++k) {
    // `origin` is the coordinate of sample `0` and the fit is in units of samples, so this is a
    // plain sum; the half-voxel offset to voxel centers is already carried by `origin`.
    fitted[displayDimensionIndices[k]] = sampled.origin[k] + center[k];
  }
  return fitted;
}
