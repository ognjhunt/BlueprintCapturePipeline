import type { SegmentationMask } from './segmentation.js';
/** Internal encoder used to create deterministic synthetic conformance data. */
export declare function encodeSegmentationMask(raster: Uint8Array, width: number, height: number): SegmentationMask;
/**
 * Strictly validates and decodes one complete SAM 3 segmentation mask. Structural
 * checks enforce the supported encodings, positive JavaScript-safe dimensions, a
 * safe decoded area, packed payload shape, and decoder finalization without a
 * project-defined quota ceiling. The returned raster is row-major and contains
 * only 0 and 1 values.
 */
export declare function decodeMaskToRaster(mask: SegmentationMask): Uint8Array;
