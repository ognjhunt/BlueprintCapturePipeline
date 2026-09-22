export interface OutputTextLane {
    readonly item_id: string;
    readonly output_index: number;
    readonly content_index: number;
}
export type ResponsesEvent = ({
    readonly type: 'response.output_text.delta';
    readonly delta: string;
} & OutputTextLane) | ({
    readonly type: 'response.output_text.done';
    readonly text: string;
} & OutputTextLane) | ({
    readonly type: 'response.refusal.delta';
    readonly delta: string;
} & OutputTextLane) | ({
    readonly type: 'response.refusal.done';
    readonly refusal: string;
} & OutputTextLane) | {
    readonly type: 'response.completed';
} | {
    readonly type: 'response.incomplete';
    readonly response: {
        readonly incomplete_details?: {
            readonly reason?: string | null;
        } | null;
    };
} | {
    readonly type: 'response.failed';
    readonly response: {
        readonly error?: {
            readonly message: string;
        } | null;
    };
} | {
    readonly type: 'error';
    readonly message: string;
};
export type ResponseStreamOutcome = {
    readonly status: 'completed';
} | {
    readonly status: 'incomplete';
    readonly reason: 'response' | 'eof';
    readonly detail?: string;
};
export interface ResponseFormatParser<Event, Result> {
    push(chunk: string, options?: {
        readonly emit?: boolean;
    }): readonly Event[];
    finish(outcome: ResponseStreamOutcome): {
        readonly events?: readonly Event[];
        readonly result: Result;
    };
}
export interface ResponseFormat<Event, Result> {
    createParser(): ResponseFormatParser<Event, Result>;
}
export interface ParsedResponsesStream<Event, Result> extends AsyncIterable<Event> {
    readonly finalResult: Promise<Result>;
}
