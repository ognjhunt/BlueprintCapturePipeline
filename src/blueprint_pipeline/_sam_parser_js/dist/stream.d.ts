import type { ParsedResponsesStream, ResponseFormat } from './types.js';
export declare function parseResponsesStream<SourceEventType extends {
    readonly type: string;
}, Event, Result>(source: AsyncIterable<SourceEventType>, format: ResponseFormat<Event, Result>): ParsedResponsesStream<Event, Result>;
