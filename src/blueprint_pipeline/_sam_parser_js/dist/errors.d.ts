import type { OutputTextLane } from './types.js';
export interface ResponsesEventReference {
    readonly type: string;
}
export declare class ResponsesStreamError extends Error {
    readonly code: string;
    constructor(message: string, code: string, options?: ErrorOptions);
}
export declare class ResponsesStreamConsumedError extends ResponsesStreamError {
    constructor();
}
export declare class ResponsesStreamAbortedError extends ResponsesStreamError {
    constructor();
}
export declare class ResponsesStreamFailedError extends ResponsesStreamError {
    readonly event: ResponsesEventReference;
    constructor(message: string, event: ResponsesEventReference);
}
export declare class ResponsesStreamEventError extends ResponsesStreamError {
    readonly event: ResponsesEventReference;
    constructor(message: string, event: ResponsesEventReference);
}
export declare class ResponsesStreamLaneError extends ResponsesStreamError {
    readonly expected: OutputTextLane | undefined;
    readonly received: OutputTextLane | undefined;
    constructor(message: string, expected: OutputTextLane | undefined, received: OutputTextLane | undefined);
}
export declare class ResponsesStreamRefusalError extends ResponsesStreamError {
    readonly event: ResponsesEventReference;
    constructor(message: string, event: ResponsesEventReference);
}
export declare class ResponsesStreamParserError extends ResponsesStreamError {
    constructor(cause: unknown);
}
export declare class ResponsesStreamSourceError extends ResponsesStreamError {
    readonly operation: 'iterator' | 'next' | 'return';
    readonly priorError?: unknown | undefined;
    constructor(operation: 'iterator' | 'next' | 'return', cause: unknown, priorError?: unknown | undefined);
}
export declare class InvalidSegmentationMaskError extends ResponsesStreamError {
    constructor(message: string, options?: ErrorOptions);
}
