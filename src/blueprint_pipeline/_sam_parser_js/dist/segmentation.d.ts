import type { ParsedResponsesStream, ResponseFormat, ResponseStreamOutcome } from './types.js';
export interface FrameReference {
    readonly frameIndex: number;
}
/** The wire encodings of a mask payload: `~` selects `lossless`, `!` selects `one_bit`. */
export type SegmentationMaskEncoding = 'lossless' | 'one_bit';
/**
 * One complete mask exactly as the SAM API emitted it. `payload` is the base85
 * text after the encoding marker and is never partial; `width` and `height` are
 * the raster's own dimensions, not the frame's.
 */
export interface SegmentationMask {
    readonly encoding: SegmentationMaskEncoding;
    readonly payload: string;
    readonly width: number;
    readonly height: number;
}
interface RecordBase {
    readonly order: number;
    readonly objectId: string;
    readonly frame?: FrameReference;
}
export interface SegmentationTextRecord {
    readonly kind: 'text';
    readonly order: number;
    readonly text: string;
}
export interface SegmentationBoxRecord extends RecordBase {
    readonly kind: 'box';
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}
export interface SegmentationMaskBounds {
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}
export interface SegmentationMaskRecord extends RecordBase {
    readonly kind: 'mask';
    readonly identity: string;
    readonly revision: number;
    readonly mask: SegmentationMask;
    /** The half-open source-pixel box the raster covers; the record's box. */
    readonly bounds: SegmentationMaskBounds;
}
export type SegmentationRecord = SegmentationTextRecord | SegmentationBoxRecord | SegmentationMaskRecord;
export type SegmentationRecordKind = SegmentationRecord['kind'];
export type SegmentationRecordOfKind<Kind extends SegmentationRecordKind> = Extract<SegmentationRecord, {
    readonly kind: Kind;
}>;
export interface SegmentationDiagnostic {
    readonly severity: 'warning' | 'error';
    readonly code: string;
    readonly message: string;
    readonly line: number;
    readonly raw: string;
}
interface SegmentationViewBase {
    readonly revision: number;
    readonly records: readonly SegmentationRecord[];
    readonly diagnostics: readonly SegmentationDiagnostic[];
    readonly rawOutput: string;
}
export interface ImageSegmentationSnapshot extends SegmentationViewBase {
    readonly media: 'image';
}
export interface VideoSegmentationSnapshot extends SegmentationViewBase {
    readonly media: 'video';
}
export type SegmentationSnapshot = ImageSegmentationSnapshot | VideoSegmentationSnapshot;
export interface ImageSegmentationResult extends ImageSegmentationSnapshot {
    readonly outcome: ResponseStreamOutcome;
}
export interface VideoSegmentationResult extends VideoSegmentationSnapshot {
    readonly outcome: ResponseStreamOutcome;
}
export type SegmentationResult = ImageSegmentationResult | VideoSegmentationResult;
declare function imageSegmentationFormat(): ResponseFormat<ImageSegmentationSnapshot, ImageSegmentationResult>;
declare function videoSegmentationFormat(): ResponseFormat<VideoSegmentationSnapshot, VideoSegmentationResult>;
export declare const formats: Readonly<{
    segmentation: Readonly<{
        image: typeof imageSegmentationFormat;
        video: typeof videoSegmentationFormat;
    }>;
}>;
export declare function parseImageStream<SourceEventType extends {
    readonly type: string;
}>(source: AsyncIterable<SourceEventType>): ParsedResponsesStream<ImageSegmentationSnapshot, ImageSegmentationResult>;
export declare function parseVideoStream<SourceEventType extends {
    readonly type: string;
}>(source: AsyncIterable<SourceEventType>): ParsedResponsesStream<VideoSegmentationSnapshot, VideoSegmentationResult>;
export declare function frameIndexOf(record: SegmentationRecord): number | undefined;
export declare function recordsOfKind<Kind extends SegmentationRecordKind>(records: readonly SegmentationRecord[], kind: Kind): readonly SegmentationRecordOfKind<Kind>[];
export {};
