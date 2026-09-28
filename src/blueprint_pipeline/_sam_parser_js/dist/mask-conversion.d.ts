import type { SegmentationMask } from './segmentation.js';
import { type RLEObject } from './coco-rle.js';
export declare function decodeMaskToRLE(mask: SegmentationMask): RLEObject;
export declare function decodeMaskToSVGPath(mask: SegmentationMask): string;
