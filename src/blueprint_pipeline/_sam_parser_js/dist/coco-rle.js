/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 *
 * Portions of this file are a TypeScript port of the COCO API mask module
 * (`common/maskApi.c` and `PythonAPI/pycocotools/_mask.pyx`).
 * Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin. All rights reserved.
 * Licensed under the Simplified BSD License; see THIRD_PARTY_NOTICES.md in
 * the repository root for the full license text.
 */
export class DataArray {
    data;
    shape;
    constructor(data, shape) {
        this.data = data;
        this.shape = shape;
    }
    reshape(shape) {
        return new DataArray(this.data, shape);
    }
}
function rleInit(R, h, w, m, cnts) {
    R.h = h;
    R.w = w;
    R.m = m;
    R.cnts = m === 0 ? [0] : cnts;
}
function rlesInit(R, n) {
    for (let i = 0; i < n; i++) {
        R[i] = { h: 0, w: 0, m: 0, cnts: [0] };
        rleInit(R[i], 0, 0, 0, [0]);
    }
}
class RLEs {
    R;
    n;
    constructor(n) {
        this.R = [];
        rlesInit(this.R, n);
        this.n = n;
    }
}
class Masks {
    mask;
    h;
    w;
    n;
    constructor(h, w, n) {
        this.mask = new Uint8Array(h * w * n);
        this.h = h;
        this.w = w;
        this.n = n;
    }
    toDataArray() {
        return new DataArray(this.mask, [this.h, this.w, this.n]);
    }
}
export function encode(bitmask) {
    if (bitmask.shape.length === 3) {
        return _encode(bitmask);
    }
    else if (bitmask.shape.length === 2) {
        const h = bitmask.shape[0];
        const w = bitmask.shape[1];
        const result = _encode(bitmask.reshape([h, w, 1]));
        return Array.isArray(result) ? result[0] : result;
    }
    throw new Error('wrong shape of bitmask');
}
function _encode(bitmask) {
    const h = bitmask.shape[0];
    const w = bitmask.shape[1];
    const n = bitmask.shape[2];
    const Rs = new RLEs(n);
    rleEncode(Rs.R, bitmask.data, h, w, n);
    return _toString(Rs);
}
export function decode(rleObjs) {
    const Rs = _frString(rleObjs);
    const h = Rs.R[0].h;
    const w = Rs.R[0].w;
    const n = Rs.n;
    const masks = new Masks(h, w, n);
    rleDecode(Rs.R, masks.mask, n);
    const dataArray = masks.toDataArray();
    return Array.isArray(rleObjs) ? dataArray : dataArray.reshape([h, w]);
}
function rleEncode(R, M, h, w, n) {
    const a = w * h;
    const cnts = [];
    for (let i = 0; i < n; i++) {
        const from = a * i;
        const to = a * (i + 1);
        const T = M.slice(from, to);
        let k = 0;
        let p = 0;
        let c = 0;
        for (let j = 0; j < a; j++) {
            if (T[j] !== p) {
                cnts[k++] = c;
                c = 0;
                p = T[j];
            }
            c++;
        }
        cnts[k++] = c;
        rleInit(R[i], h, w, k, [...cnts]);
    }
}
function rleDecode(R, M, n) {
    let p = 0;
    for (let i = 0; i < n; i++) {
        let v = false;
        const rle = R[i];
        for (let j = 0; j < rle.m; j++) {
            for (let k = 0; k < rle.cnts[j]; k++) {
                M[p++] = v === false ? 0 : 1;
            }
            v = !v;
        }
    }
}
function rleToString(R) {
    const m = R.m;
    let p = 0;
    const s = [];
    for (let i = 0; i < m; i++) {
        let x = R.cnts[i];
        if (i > 2) {
            x -= R.cnts[i - 2];
        }
        let more = true;
        while (more) {
            let c = x & 0x1f;
            x >>= 5;
            more = c & 0x10 ? x != -1 : x != 0;
            if (more) {
                c |= 0x20;
            }
            c += 48;
            s[p++] = String.fromCharCode(c);
        }
    }
    return s.join('');
}
function _toString(Rs) {
    const n = Rs.n;
    const objs = [];
    for (let i = 0; i < n; i++) {
        const rle = Rs.R[i];
        const c_string = rleToString(rle);
        objs.push({
            size: [rle.h, rle.w],
            counts: c_string,
        });
    }
    return objs;
}
function _frString(inRleObjs) {
    const rleObjs = Array.isArray(inRleObjs) ? inRleObjs : [inRleObjs];
    const n = rleObjs.length;
    const Rs = new RLEs(n);
    for (let i = 0; i < rleObjs.length; i++) {
        const obj = rleObjs[i];
        rleFrString(Rs.R[i], obj.counts, obj.size[0], obj.size[1]);
    }
    return Rs;
}
function rleFrString(R, s, h, w) {
    let m = 0;
    let p = 0;
    const cnts = [];
    while (s[p]) {
        let x = 0;
        let k = 0;
        let more = 1;
        while (more) {
            const c = s.charCodeAt(p) - 48;
            x |= (c & 0x1f) << (5 * k);
            more = c & 0x20;
            p++;
            k++;
            if (!more && c & 0x10) {
                x |= -1 << (5 * k);
            }
        }
        if (m > 2) {
            x += cnts[m - 2];
        }
        cnts[m++] = x;
    }
    rleInit(R, h, w, m, cnts);
}
