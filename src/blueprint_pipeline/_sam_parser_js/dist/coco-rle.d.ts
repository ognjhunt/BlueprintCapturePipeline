export declare class DataArray {
    data: Uint8Array;
    shape: ReadonlyArray<number>;
    constructor(data: Uint8Array, shape: Array<number>);
    reshape(shape: Array<number>): DataArray;
}
export type RLEObject = {
    size: [height: number, width: number];
    counts: string;
};
export declare function encode(bitmask: DataArray): RLEObject | Array<RLEObject>;
export declare function decode(rleObjs: RLEObject | Array<RLEObject>): DataArray;
