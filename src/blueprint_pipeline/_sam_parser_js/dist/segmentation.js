/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 */
import { InvalidSegmentationMaskError } from './errors.js';
import { decodeMaskToRaster } from './mask-codec.js';
import { parseResponsesStream } from './stream.js';
/**
 * One SAM API record: an ASCII-decimal object id, one box token, one mask token.
 * The payload alphabet never contains `|`, so `[^|]+` ends exactly at the mask's
 * closing `|>`; `,` before a record is the wire separator and is optional.
 */
const apiRecordPattern = /^(?:,)?([0-9]+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)\|><\|mask;x=0;y=0;data=(\d+),(\d+),([!~][^|]+)\|>/;
function freezeFrame(frameIndex) {
    return frameIndex === undefined ? undefined : Object.freeze({ frameIndex });
}
function freezeRecord(record) {
    if (record.kind === 'mask')
        Object.freeze(record.mask);
    if ('frame' in record && record.frame !== undefined)
        Object.freeze(record.frame);
    return Object.freeze(record);
}
class SegmentationParser {
    #media;
    #records = [];
    #diagnostics = [];
    #revisions = new Map();
    #rawOutput = '';
    #bufferParts = [];
    #line = 0;
    #revision = 0;
    constructor(media) {
        this.#media = media;
    }
    push(chunk, options = {}) {
        if (typeof chunk !== 'string')
            throw new TypeError('Parser chunks must be strings.');
        this.#rawOutput += chunk;
        let changed = false;
        const segments = chunk.split('\n');
        for (const segment of segments.slice(0, -1)) {
            this.#appendBuffer(segment);
            const buffered = this.#bufferParts.join('');
            const line = buffered.endsWith('\r') ? buffered.slice(0, -1) : buffered;
            this.#clearBuffer();
            const priorRecords = this.#records.length;
            const priorDiagnostics = this.#diagnostics.length;
            this.#acceptLine(line);
            changed =
                changed ||
                    priorRecords !== this.#records.length ||
                    priorDiagnostics !== this.#diagnostics.length;
        }
        this.#appendBuffer(segments.at(-1));
        if (changed)
            this.#revision += 1;
        return changed && options.emit !== false ? [this.#snapshot()] : [];
    }
    finish(outcome) {
        const events = [];
        if (this.#bufferParts.length > 0) {
            const buffered = this.#bufferParts.join('');
            const line = buffered.endsWith('\r') ? buffered.slice(0, -1) : buffered;
            this.#clearBuffer();
            const priorRecords = this.#records.length;
            const priorDiagnostics = this.#diagnostics.length;
            this.#acceptLine(line);
            if (priorRecords !== this.#records.length ||
                priorDiagnostics !== this.#diagnostics.length) {
                this.#revision += 1;
                events.push(this.#snapshot());
            }
        }
        const view = this.#view();
        return {
            events,
            result: Object.freeze({
                ...view,
                outcome: Object.freeze({ ...outcome }),
            }),
        };
    }
    #appendBuffer(value) {
        if (value.length > 0)
            this.#bufferParts.push(value);
    }
    #clearBuffer() {
        this.#bufferParts.length = 0;
    }
    #acceptLine(raw) {
        this.#line += 1;
        const line = raw.trim();
        if (line.length === 0)
            return false;
        if (line.startsWith('<')) {
            const accepted = this.#acceptApiLine(line, raw);
            if (accepted !== undefined)
                return accepted;
        }
        this.#addRecord(freezeRecord({ kind: 'text', order: this.#records.length, text: raw }));
        return true;
    }
    #acceptApiLine(line, raw) {
        const header = /^<(\d+)f>(.*)$/.exec(line);
        if (header === null)
            return undefined;
        const frameIndex = Number(header[1]);
        if (!Number.isSafeInteger(frameIndex) || frameIndex < 0) {
            this.#diagnose('invalid_frame', 'Frame references must be non-negative safe integers.', raw);
            return true;
        }
        if (this.#media === 'image' && frameIndex !== 0) {
            this.#diagnose('unexpected_frame', 'Image segmentation records require frame zero.', raw);
            return true;
        }
        const frame = this.#media === 'video' ? freezeFrame(frameIndex) : undefined;
        let remainder = header[2];
        if (remainder.length === 0) {
            this.#diagnose('malformed_record', 'Malformed SAM API object record.', raw);
            return true;
        }
        let accepted = false;
        while (remainder.length > 0) {
            const match = apiRecordPattern.exec(remainder);
            if (match === null) {
                this.#diagnose('malformed_record', 'Malformed SAM API object record.', raw);
                return true;
            }
            const objectId = match[1];
            const [left, top, inclusiveRight, inclusiveBottom, sourceWidth, sourceHeight] = match.slice(2, 8).map((value) => {
                const parsed = Number(value);
                return parsed === 0 ? 0 : parsed;
            });
            const maskHeight = Number(match[8]);
            const maskWidth = Number(match[9]);
            const payload = match[10];
            if (![left, top, inclusiveRight, inclusiveBottom, sourceWidth, sourceHeight].every(Number.isSafeInteger) ||
                sourceWidth <= 0 ||
                sourceHeight <= 0 ||
                left < 0 ||
                top < 0 ||
                inclusiveRight < left ||
                inclusiveBottom < top ||
                inclusiveRight >= sourceWidth ||
                inclusiveBottom >= sourceHeight) {
                this.#diagnose('invalid_box', 'SAM API box coordinates are invalid.', raw);
                return true;
            }
            const bounds = Object.freeze({
                left,
                top,
                right: inclusiveRight + 1,
                bottom: inclusiveBottom + 1,
            });
            this.#addRecord(freezeRecord({
                kind: 'box',
                order: this.#records.length,
                objectId,
                ...(frame === undefined ? {} : { frame }),
                ...bounds,
            }));
            this.#acceptMask(objectId, frame, {
                encoding: payload.startsWith('~') ? 'lossless' : 'one_bit',
                payload,
                width: maskWidth,
                height: maskHeight,
            }, raw, bounds);
            accepted = true;
            remainder = remainder.slice(match[0].length);
        }
        return accepted;
    }
    #acceptMask(objectId, frame, mask, raw, bounds) {
        const { width, height } = mask;
        if (width <= 0 || height <= 0 || !Number.isSafeInteger(width * height)) {
            this.#diagnose('invalid_mask_size', 'Mask dimensions must be positive.', raw);
            return;
        }
        try {
            decodeMaskToRaster(mask);
        }
        catch (error) {
            if (!(error instanceof InvalidSegmentationMaskError))
                throw error;
            this.#diagnose('invalid_mask_payload', error.message, raw);
            return;
        }
        const identity = `${this.#media}:${frame?.frameIndex ?? '*'}:${objectId}`;
        const revision = (this.#revisions.get(identity) ?? 0) + 1;
        this.#revisions.set(identity, revision);
        this.#addRecord(freezeRecord({
            kind: 'mask',
            order: this.#records.length,
            objectId,
            ...(frame === undefined ? {} : { frame }),
            identity,
            revision,
            mask: Object.freeze({ ...mask }),
            bounds,
        }));
    }
    #addRecord(record) {
        this.#records.push(record);
    }
    #diagnose(code, message, raw) {
        this.#diagnostics.push(Object.freeze({ severity: 'error', code, message, line: this.#line, raw }));
    }
    #view() {
        const common = {
            revision: this.#revision,
            records: Object.freeze([...this.#records]),
            diagnostics: Object.freeze([...this.#diagnostics]),
            rawOutput: this.#rawOutput,
        };
        return Object.freeze({
            media: this.#media,
            ...common,
        });
    }
    #snapshot() {
        return this.#view();
    }
}
function segmentationFormat(media) {
    return Object.freeze({
        createParser: () => new SegmentationParser(media),
    });
}
function imageSegmentationFormat() {
    if (arguments.length !== 0) {
        throw new TypeError('formats.segmentation.image() does not accept arguments.');
    }
    return segmentationFormat('image');
}
function videoSegmentationFormat() {
    if (arguments.length !== 0) {
        throw new TypeError('formats.segmentation.video() does not accept arguments.');
    }
    return segmentationFormat('video');
}
const segmentation = Object.freeze({
    image: imageSegmentationFormat,
    video: videoSegmentationFormat,
});
export const formats = Object.freeze({ segmentation });
export function parseImageStream(source) {
    return parseResponsesStream(source, segmentation.image());
}
export function parseVideoStream(source) {
    return parseResponsesStream(source, segmentation.video());
}
export function frameIndexOf(record) {
    return record.kind === 'text' ? undefined : record.frame?.frameIndex;
}
export function recordsOfKind(records, kind) {
    return Object.freeze(records.filter((record) => record.kind === kind));
}
