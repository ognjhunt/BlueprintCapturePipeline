/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 */
import { InvalidSegmentationMaskError } from './errors.js';
const excluded = new Set(['"', '\\', ',', ';', '<', '>', '|']);
const alphabet = (() => {
    let value = '';
    for (let code = 0x21; code <= 0x7e; code += 1) {
        const character = String.fromCharCode(code);
        if (!excluded.has(character))
            value += character;
    }
    return value;
})();
const radix = 85;
const prefixLength = 5;
const characterValues = new Int16Array(128).fill(-1);
for (let index = 0; index < alphabet.length; index += 1) {
    characterValues[alphabet.charCodeAt(index)] = index;
}
const top = 1 << 24;
const bottom = 1 << 16;
const mask32 = 0xffffffff;
const increment = 14;
const countLimit = 4096;
const contextCount = 1 << 12;
const padding = 2;
const symbols = 256;
const totalIndex = symbols;
const spatialModes = 5;
const zeroIncrement = 32;
const zeroLimit = 16384;
const orderOneIncrement = 56;
const orderOneLimit = 8192;
const orderTwoStep = 14 * 44;
const orderTwoLimit = 14 * 3584;
const orderZeroStep = 2 * 16;
const orderZeroLimit = 2 * 2048;
const orderZeroInitial = 2;
const orderOneInitial = 1;
function invalid(message, cause) {
    return new InvalidSegmentationMaskError(message, cause === undefined ? undefined : { cause });
}
function digit(character) {
    const code = character.charCodeAt(0);
    const value = code < characterValues.length ? (characterValues[code] ?? -1) : -1;
    if (value < 0 || value >= radix) {
        throw invalid('Mask payload contains an invalid character.');
    }
    return value;
}
function pack(input) {
    let length = input.length;
    const prefix = new Array(prefixLength);
    for (let index = prefixLength - 1; index >= 0; index -= 1) {
        prefix[index] = alphabet[length % radix];
        length = Math.floor(length / radix);
    }
    if (length !== 0)
        throw invalid('Mask payload is too large.');
    const output = [...prefix];
    let offset = 0;
    const full = input.length - (input.length % 4);
    for (; offset < full; offset += 4) {
        let value = ((input[offset] << 24) |
            (input[offset + 1] << 16) |
            (input[offset + 2] << 8) |
            input[offset + 3]) >>>
            0;
        const digits = new Array(5);
        for (let index = 4; index >= 0; index -= 1) {
            digits[index] = value % radix;
            value = (value - digits[index]) / radix;
        }
        for (const encoded of digits)
            output.push(alphabet[encoded]);
    }
    const remainder = input.length - offset;
    if (remainder > 0) {
        let value = 0;
        for (let index = 0; index < 4; index += 1) {
            value = ((value << 8) | (index < remainder ? input[offset + index] : 0)) >>> 0;
        }
        const digits = new Array(5);
        for (let index = 4; index >= 0; index -= 1) {
            digits[index] = value % radix;
            value = (value - digits[index]) / radix;
        }
        for (let index = 0; index < remainder + 1; index += 1) {
            output.push(alphabet[digits[index]]);
        }
    }
    return output.join('');
}
function unpack(payload) {
    if (payload.length < prefixLength) {
        throw invalid('Mask payload is missing its length prefix.');
    }
    let length = 0;
    for (let index = 0; index < prefixLength; index += 1) {
        length = length * radix + digit(payload[index]);
    }
    if (!Number.isSafeInteger(length))
        throw invalid('Mask payload length is unsupported.');
    const remainder = length % 4;
    const expected = prefixLength + Math.floor(length / 4) * 5 + (remainder === 0 ? 0 : remainder + 1);
    if (payload.length !== expected) {
        throw invalid('Mask payload length does not match its prefix.');
    }
    const output = new Uint8Array(length);
    let source = prefixLength;
    let destination = 0;
    const full = length - remainder;
    for (; destination < full; destination += 4) {
        let value = 0;
        for (let index = 0; index < 5; index += 1) {
            value = value * radix + digit(payload[source + index]);
        }
        if (value > mask32)
            throw invalid('Mask payload contains an out-of-range group.');
        source += 5;
        output[destination] = (value >>> 24) & 0xff;
        output[destination + 1] = (value >>> 16) & 0xff;
        output[destination + 2] = (value >>> 8) & 0xff;
        output[destination + 3] = value & 0xff;
    }
    if (remainder > 0) {
        let value = 0;
        for (let index = 0; index < 5; index += 1) {
            value =
                value * radix +
                    (index < remainder + 1 ? digit(payload[source + index]) : radix - 1);
        }
        if (value > mask32)
            throw invalid('Mask payload contains an out-of-range tail.');
        for (let index = 0; index < remainder; index += 1) {
            output[destination + index] = (value >>> (24 - index * 8)) & 0xff;
        }
    }
    if (pack(output) !== payload) {
        throw invalid('Mask payload is not in canonical form.');
    }
    return output;
}
class RangeEncoder {
    #low = 0;
    #range = mask32;
    #output = [];
    encode(cumulative, frequency, total) {
        const scaled = Math.floor(this.#range / total);
        this.#low = (this.#low + scaled * cumulative) >>> 0;
        this.#range = scaled * frequency;
        while ((this.#low ^ (this.#low + this.#range)) >>> 0 < top ||
            (this.#range < bottom &&
                ((this.#range = (-this.#low >>> 0) & (bottom - 1)), true))) {
            this.#output.push((this.#low >>> 24) & 0xff);
            this.#low = (this.#low << 8) >>> 0;
            this.#range = (this.#range << 8) >>> 0;
        }
    }
    finish() {
        for (let index = 0; index < 4; index += 1) {
            this.#output.push((this.#low >>> 24) & 0xff);
            this.#low = (this.#low << 8) >>> 0;
        }
        return Uint8Array.from(this.#output);
    }
}
class RangeDecoder {
    #input;
    #position = 0;
    #low = 0;
    #range = mask32;
    #code = 0;
    #scaled = 0;
    constructor(input) {
        if (input.length < 4)
            throw invalid('Mask payload is missing its finalization.');
        this.#input = input;
        for (let index = 0; index < 4; index += 1) {
            this.#code = ((this.#code << 8) | this.#read()) >>> 0;
        }
    }
    frequency(total) {
        this.#scaled = Math.floor(this.#range / total);
        if (this.#scaled === 0)
            throw invalid('Mask payload has invalid coding state.');
        const value = Math.floor(((this.#code - this.#low) >>> 0) / this.#scaled);
        return value >= total ? total - 1 : value;
    }
    getFreq(total) {
        return this.frequency(total);
    }
    decode(cumulative, frequency) {
        this.#low = (this.#low + this.#scaled * cumulative) >>> 0;
        this.#range = this.#scaled * frequency;
        while ((this.#low ^ (this.#low + this.#range)) >>> 0 < top ||
            (this.#range < bottom &&
                ((this.#range = (-this.#low >>> 0) & (bottom - 1)), true))) {
            this.#code = ((this.#code << 8) | this.#read()) >>> 0;
            this.#low = (this.#low << 8) >>> 0;
            this.#range = (this.#range << 8) >>> 0;
        }
    }
    #read() {
        if (this.#position >= this.#input.length) {
            throw invalid('Mask payload ended before decoding completed.');
        }
        return this.#input[this.#position++];
    }
}
function getOrderTwo(map, key) {
    let counts = map.get(key);
    if (counts === undefined) {
        counts = new Uint16Array(symbols + 1);
        map.set(key, counts);
    }
    return counts;
}
function paeth(left, up, upperLeft) {
    const estimate = left + up - upperLeft;
    const leftDistance = Math.abs(estimate - left);
    const upDistance = Math.abs(estimate - up);
    const upperLeftDistance = Math.abs(estimate - upperLeft);
    if (leftDistance <= upDistance && leftDistance <= upperLeftDistance)
        return left;
    return upDistance <= upperLeftDistance ? up : upperLeft;
}
function predictByte(output, offset, x, y, mode, width) {
    if (mode < 0 || mode >= spatialModes) {
        throw invalid('Lossless mask payload uses an invalid predictor.');
    }
    const left = x > 0 ? output[offset - 1] : 0;
    const up = y > 0 ? output[offset - width] : 0;
    if (mode === 1)
        return up;
    if (mode === 2)
        return left;
    if (mode === 3)
        return (left + up) >> 1;
    const upperLeft = x > 0 && y > 0 ? output[offset - width - 1] : 0;
    if (mode === 0)
        return paeth(left, up, upperLeft);
    if (upperLeft >= Math.max(left, up))
        return Math.min(left, up);
    if (upperLeft <= Math.min(left, up))
        return Math.max(left, up);
    return left + up - upperLeft;
}
function unzig(value, prediction) {
    const delta = value & 1 ? -((value + 1) >> 1) : value >> 1;
    return (prediction + delta) & 0xff;
}
function zeroContext(residuals, offset, x, y, width) {
    const left = x > 0 ? (residuals[offset - 1] === 0 ? 1 : 0) : 1;
    const up = y > 0 ? (residuals[offset - width] === 0 ? 1 : 0) : 1;
    const upperLeft = x > 0 && y > 0 ? (residuals[offset - width - 1] === 0 ? 1 : 0) : 1;
    const upperRight = y > 0 && x < width - 1 ? (residuals[offset - width + 1] === 0 ? 1 : 0) : 1;
    return left | (up << 1) | (upperLeft << 2) | (upperRight << 3);
}
function decodeLosslessRaster(payload, width, height) {
    if (!payload.startsWith('~')) {
        throw invalid('lossless mask payloads must start with ~.');
    }
    const packed = unpack(payload.slice(1));
    if (packed.length < 5)
        throw invalid('Lossless mask payload is truncated.');
    const selector = packed[0];
    const decoder = new RangeDecoder(packed.subarray(1));
    const length = width * height;
    const output = new Uint8Array(length);
    const residuals = new Uint8Array(length);
    const zeroCounts = new Uint32Array(32).fill(1);
    const orderOne = new Uint16Array(symbols * symbols).fill(orderOneInitial);
    const orderOneTotals = new Int32Array(symbols).fill(symbols * orderOneInitial);
    const orderZero = new Uint16Array(symbols).fill(orderZeroInitial);
    let orderZeroTotal = symbols * orderZeroInitial;
    const orderTwo = new Map();
    let previous = 0;
    for (let offset = 0; offset < length; offset += 1) {
        const x = offset % width;
        const y = Math.floor(offset / width);
        const zeroOffset = zeroContext(residuals, offset, x, y, width) << 1;
        const zero = zeroCounts[zeroOffset];
        const nonzero = zeroCounts[zeroOffset + 1];
        const zeroFrequency = decoder.getFreq(zero + nonzero);
        const bit = zeroFrequency < zero ? 0 : 1;
        decoder.decode(bit === 0 ? 0 : zero, bit === 0 ? zero : nonzero);
        zeroCounts[zeroOffset + bit] = zeroCounts[zeroOffset + bit] + zeroIncrement;
        if (zeroCounts[zeroOffset] + zeroCounts[zeroOffset + 1] >= zeroLimit) {
            zeroCounts[zeroOffset] = zeroCounts[zeroOffset] >> 1 || 1;
            zeroCounts[zeroOffset + 1] = zeroCounts[zeroOffset + 1] >> 1 || 1;
        }
        let symbol = 0;
        if (bit === 1) {
            const above = offset >= width ? residuals[offset - width] : 0;
            const second = getOrderTwo(orderTwo, (previous << 8) | above);
            const firstOffset = previous << 8;
            const excludedZero = second[0] + orderOne[firstOffset] + orderZero[0];
            const total = second[totalIndex] + orderOneTotals[previous] + orderZeroTotal - excludedZero;
            const target = decoder.getFreq(total);
            let cumulative = 0;
            symbol = 1;
            let frequency = second[1] + orderOne[firstOffset + 1] + orderZero[1];
            while (cumulative + frequency <= target) {
                cumulative += frequency;
                symbol += 1;
                if (symbol >= symbols)
                    throw invalid('Lossless mask payload is malformed.');
                frequency =
                    second[symbol] + orderOne[firstOffset + symbol] + orderZero[symbol];
            }
            decoder.decode(cumulative, frequency);
            second[symbol] = second[symbol] + orderTwoStep;
            second[totalIndex] = second[totalIndex] + orderTwoStep;
            if (second[totalIndex] >= orderTwoLimit) {
                let totalAfterScaling = 0;
                for (let value = 0; value < symbols; value += 1) {
                    second[value] = second[value] >> 1;
                    totalAfterScaling += second[value];
                }
                second[totalIndex] = totalAfterScaling;
            }
            orderOne[firstOffset + symbol] =
                orderOne[firstOffset + symbol] + orderOneIncrement;
            orderOneTotals[previous] = orderOneTotals[previous] + orderOneIncrement;
            if (orderOneTotals[previous] >= orderOneLimit) {
                let totalAfterScaling = 0;
                for (let value = 0; value < symbols; value += 1) {
                    const scaled = orderOne[firstOffset + value] >> 1 || 1;
                    orderOne[firstOffset + value] = scaled;
                    totalAfterScaling += scaled;
                }
                orderOneTotals[previous] = totalAfterScaling;
            }
            orderZero[symbol] = orderZero[symbol] + orderZeroStep;
            orderZeroTotal += orderZeroStep;
            if (orderZeroTotal >= orderZeroLimit) {
                let totalAfterScaling = 0;
                for (let value = 0; value < symbols; value += 1) {
                    orderZero[value] = orderZero[value] >> 1;
                    totalAfterScaling += orderZero[value];
                }
                orderZeroTotal = totalAfterScaling;
            }
        }
        residuals[offset] = symbol;
        output[offset] = unzig(symbol, predictByte(output, offset, x, y, selector, width));
        previous = symbol;
    }
    return output;
}
function contextAt(raster, position, prior, earlier) {
    return (raster[position - 1] |
        (raster[position - 2] << 1) |
        (raster[prior - 2] << 2) |
        (raster[prior - 1] << 3) |
        (raster[prior] << 4) |
        (raster[prior + 1] << 5) |
        (raster[prior + 2] << 6) |
        (raster[earlier - 2] << 7) |
        (raster[earlier - 1] << 8) |
        (raster[earlier] << 9) |
        (raster[earlier + 1] << 10) |
        (raster[earlier + 2] << 11));
}
function updateCounts(counts, offset, bit) {
    counts[offset + bit] = counts[offset + bit] + increment;
    if (counts[offset] + counts[offset + 1] >= countLimit) {
        counts[offset] = counts[offset] >> 1 || 1;
        counts[offset + 1] = counts[offset + 1] >> 1 || 1;
    }
}
function encodeRaster(input, width, height) {
    const encoder = new RangeEncoder();
    const counts = new Uint16Array(contextCount * 2).fill(1);
    const paddedWidth = width + 2 * padding;
    const padded = new Uint8Array(paddedWidth * (height + padding));
    for (let y = 0; y < height; y += 1) {
        const row = (y + padding) * paddedWidth + padding;
        for (let x = 0; x < width; x += 1) {
            const position = row + x;
            const context = contextAt(padded, position, position - paddedWidth, position - 2 * paddedWidth);
            const offset = context << 1;
            const zero = counts[offset];
            const one = counts[offset + 1];
            const bit = input[y * width + x];
            if (bit !== 0 && bit !== 1)
                throw invalid('Decoded mask is not binary.');
            if (bit === 0)
                encoder.encode(0, zero, zero + one);
            else
                encoder.encode(zero, one, zero + one);
            padded[position] = bit;
            updateCounts(counts, offset, bit);
        }
    }
    return `!${pack(encoder.finish())}`;
}
function decodeRaster(payload, width, height) {
    const decoder = new RangeDecoder(unpack(payload.slice(1)));
    const counts = new Uint16Array(contextCount * 2).fill(1);
    const paddedWidth = width + 2 * padding;
    const padded = new Uint8Array(paddedWidth * (height + padding));
    const output = new Uint8Array(width * height);
    for (let y = 0; y < height; y += 1) {
        const row = (y + padding) * paddedWidth + padding;
        for (let x = 0; x < width; x += 1) {
            const position = row + x;
            const context = contextAt(padded, position, position - paddedWidth, position - 2 * paddedWidth);
            const offset = context << 1;
            const zero = counts[offset];
            const one = counts[offset + 1];
            const value = decoder.frequency(zero + one);
            const bit = value < zero ? 0 : 1;
            if (bit === 0)
                decoder.decode(0, zero);
            else
                decoder.decode(zero, one);
            padded[position] = bit;
            output[y * width + x] = bit;
            updateCounts(counts, offset, bit);
        }
    }
    return output;
}
/** Internal encoder used to create deterministic synthetic conformance data. */
export function encodeSegmentationMask(raster, width, height) {
    if (!Number.isSafeInteger(width) ||
        !Number.isSafeInteger(height) ||
        width <= 0 ||
        height <= 0 ||
        width * height !== raster.length) {
        throw invalid('Mask raster dimensions do not match its complete payload.');
    }
    return {
        encoding: 'one_bit',
        payload: encodeRaster(raster, width, height),
        width,
        height,
    };
}
/**
 * Strictly validates and decodes one complete SAM 3 segmentation mask. Structural
 * checks enforce the supported encodings, positive JavaScript-safe dimensions, a
 * safe decoded area, packed payload shape, and decoder finalization without a
 * project-defined quota ceiling. The returned raster is row-major and contains
 * only 0 and 1 values.
 */
export function decodeMaskToRaster(mask) {
    if (mask.encoding !== 'one_bit' && mask.encoding !== 'lossless') {
        throw invalid(`Unsupported complete mask encoding: ${mask.encoding}.`);
    }
    if (!Number.isSafeInteger(mask.width) ||
        !Number.isSafeInteger(mask.height) ||
        mask.width <= 0 ||
        mask.height <= 0) {
        throw invalid('Mask dimensions must be positive safe integers.');
    }
    const area = mask.width * mask.height;
    if (!Number.isSafeInteger(area)) {
        throw invalid('Mask dimensions produce an unsafe decoded area.');
    }
    if (typeof mask.payload !== 'string') {
        throw invalid('Mask payload must be a string.');
    }
    if (mask.encoding === 'one_bit' && !mask.payload.startsWith('!')) {
        throw invalid('one_bit mask payloads must start with !.');
    }
    if (mask.encoding === 'lossless' && !mask.payload.startsWith('~')) {
        throw invalid('lossless mask payloads must start with ~.');
    }
    try {
        if (mask.encoding === 'lossless') {
            const coverage = decodeLosslessRaster(mask.payload, mask.width, mask.height);
            return coverage.map((value) => (value >= 129 ? 1 : 0));
        }
        const raster = decodeRaster(mask.payload, mask.width, mask.height);
        if (encodeRaster(raster, mask.width, mask.height) !== mask.payload) {
            throw invalid('Mask payload is not canonical or has invalid finalization.');
        }
        return raster;
    }
    catch (error) {
        if (error instanceof InvalidSegmentationMaskError)
            throw error;
        throw invalid('Mask payload could not be decoded.', error);
    }
}
