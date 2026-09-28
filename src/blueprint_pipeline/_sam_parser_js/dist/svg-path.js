/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 */
import { decode } from './coco-rle.js';
export function decodeToSVGPath(rleObjs) {
    const rleArray = Array.isArray(rleObjs) ? rleObjs : [rleObjs];
    const paths = [];
    for (const rleObj of rleArray) {
        const maskData = decode(rleObj);
        const height = rleObj.size[0];
        const width = rleObj.size[1];
        const svgPath = traceContours(maskData.data, height, width);
        paths.push(svgPath);
    }
    return paths;
}
function traceContours(mask, height, width) {
    const getPixel = (row, col) => {
        if (row < 0 || row >= height || col < 0 || col >= width) {
            return 0;
        }
        return mask[col * height + row];
    };
    const segments = [];
    for (let row = 0; row <= height; row++) {
        for (let col = 0; col <= width; col++) {
            const tl = getPixel(row - 1, col - 1);
            const tr = getPixel(row - 1, col);
            const bl = getPixel(row, col - 1);
            const br = getPixel(row, col);
            const caseIndex = (tl << 3) | (tr << 2) | (br << 1) | bl;
            const top = { x: col - 0.5, y: row - 1 };
            const bottom = { x: col - 0.5, y: row };
            const left = { x: col - 1, y: row - 0.5 };
            const right = { x: col, y: row - 0.5 };
            switch (caseIndex) {
                case 0:
                case 15:
                    break;
                case 1:
                case 14:
                    segments.push({
                        x1: left.x,
                        y1: left.y,
                        x2: bottom.x,
                        y2: bottom.y,
                    });
                    break;
                case 2:
                case 13:
                    segments.push({
                        x1: bottom.x,
                        y1: bottom.y,
                        x2: right.x,
                        y2: right.y,
                    });
                    break;
                case 3:
                case 12:
                    segments.push({
                        x1: left.x,
                        y1: left.y,
                        x2: right.x,
                        y2: right.y,
                    });
                    break;
                case 4:
                case 11:
                    segments.push({
                        x1: top.x,
                        y1: top.y,
                        x2: right.x,
                        y2: right.y,
                    });
                    break;
                case 5:
                    segments.push({
                        x1: left.x,
                        y1: left.y,
                        x2: top.x,
                        y2: top.y,
                    });
                    segments.push({
                        x1: bottom.x,
                        y1: bottom.y,
                        x2: right.x,
                        y2: right.y,
                    });
                    break;
                case 6:
                case 9:
                    segments.push({
                        x1: top.x,
                        y1: top.y,
                        x2: bottom.x,
                        y2: bottom.y,
                    });
                    break;
                case 7:
                case 8:
                    segments.push({
                        x1: left.x,
                        y1: left.y,
                        x2: top.x,
                        y2: top.y,
                    });
                    break;
                case 10:
                    segments.push({
                        x1: top.x,
                        y1: top.y,
                        x2: right.x,
                        y2: right.y,
                    });
                    segments.push({
                        x1: left.x,
                        y1: left.y,
                        x2: bottom.x,
                        y2: bottom.y,
                    });
                    break;
                default:
                    break;
            }
        }
    }
    if (segments.length === 0) {
        return '';
    }
    const paths = connectSegments(segments);
    const svgPaths = [];
    for (const path of paths) {
        if (path.length < 2) {
            continue;
        }
        const first = path[0];
        const commands = ['M' + first.x + ' ' + first.y];
        for (let i = 1; i < path.length; i++) {
            const point = path[i];
            commands.push('L' + point.x + ' ' + point.y);
        }
        commands.push('Z');
        svgPaths.push(commands.join(''));
    }
    return svgPaths.join('');
}
function connectSegments(segments) {
    const adjacency = new Map();
    const pointKey = (x, y) => `${x},${y}`;
    for (const seg of segments) {
        const key1 = pointKey(seg.x1, seg.y1);
        const key2 = pointKey(seg.x2, seg.y2);
        if (!adjacency.has(key1)) {
            adjacency.set(key1, []);
        }
        adjacency.get(key1).push({ x: seg.x2, y: seg.y2 });
        if (!adjacency.has(key2)) {
            adjacency.set(key2, []);
        }
        adjacency.get(key2).push({ x: seg.x1, y: seg.y1 });
    }
    const paths = [];
    const visited = new Set();
    for (const [startKey, neighbors] of adjacency) {
        if (visited.has(startKey) || neighbors.length === 0) {
            continue;
        }
        const path = [];
        const parts = startKey.split(',').map(Number);
        let current = { x: parts[0], y: parts[1] };
        let prevKey = null;
        while (true) {
            const currentKey = pointKey(current.x, current.y);
            path.push({ x: current.x, y: current.y });
            const currentNeighbors = adjacency.get(currentKey);
            if (currentNeighbors == null || currentNeighbors.length === 0) {
                break;
            }
            let next = null;
            for (const neighbor of currentNeighbors) {
                const neighborKey = pointKey(neighbor.x, neighbor.y);
                if (neighborKey !== prevKey) {
                    next = neighbor;
                    break;
                }
            }
            if (next == null) {
                break;
            }
            const nextKey = pointKey(next.x, next.y);
            const currentNeighborsFiltered = currentNeighbors.filter((n) => pointKey(n.x, n.y) !== nextKey);
            if (currentNeighborsFiltered.length === 0) {
                adjacency.delete(currentKey);
            }
            else {
                adjacency.set(currentKey, currentNeighborsFiltered);
            }
            const nextNeighbors = adjacency.get(nextKey);
            if (nextNeighbors != null) {
                const nextNeighborsFiltered = nextNeighbors.filter((n) => pointKey(n.x, n.y) !== currentKey);
                if (nextNeighborsFiltered.length === 0) {
                    adjacency.delete(nextKey);
                }
                else {
                    adjacency.set(nextKey, nextNeighborsFiltered);
                }
            }
            if (nextKey === startKey) {
                break;
            }
            prevKey = currentKey;
            current = next;
            visited.add(currentKey);
        }
        if (path.length >= 3) {
            paths.push(path);
        }
    }
    return paths;
}
