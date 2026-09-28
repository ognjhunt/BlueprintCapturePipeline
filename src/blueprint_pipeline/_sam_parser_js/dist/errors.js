/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 */
export class ResponsesStreamError extends Error {
    code;
    constructor(message, code, options) {
        super(message, options);
        this.code = code;
        this.name = new.target.name;
    }
}
export class ResponsesStreamConsumedError extends ResponsesStreamError {
    constructor() {
        super('A parsed response stream can be iterated only once.', 'stream_consumed');
    }
}
export class ResponsesStreamAbortedError extends ResponsesStreamError {
    constructor() {
        super('The parsed response stream was abandoned before completion.', 'stream_aborted');
    }
}
export class ResponsesStreamFailedError extends ResponsesStreamError {
    event;
    constructor(message, event) {
        super(message, 'response_failed');
        this.event = event;
    }
}
export class ResponsesStreamEventError extends ResponsesStreamError {
    event;
    constructor(message, event) {
        super(message, 'response_error');
        this.event = event;
    }
}
export class ResponsesStreamLaneError extends ResponsesStreamError {
    expected;
    received;
    constructor(message, expected, received) {
        super(message, 'response_lane');
        this.expected = expected;
        this.received = received;
    }
}
export class ResponsesStreamRefusalError extends ResponsesStreamError {
    event;
    constructor(message, event) {
        super(message, 'response_refusal');
        this.event = event;
    }
}
export class ResponsesStreamParserError extends ResponsesStreamError {
    constructor(cause) {
        super('The response format parser failed.', 'parser_error', { cause });
    }
}
export class ResponsesStreamSourceError extends ResponsesStreamError {
    operation;
    priorError;
    constructor(operation, cause, priorError) {
        super(`The Responses event source failed during ${operation}.`, 'source_error', {
            cause,
        });
        this.operation = operation;
        this.priorError = priorError;
    }
}
export class InvalidSegmentationMaskError extends ResponsesStreamError {
    constructor(message, options) {
        super(message, 'invalid_mask_payload', options);
    }
}
