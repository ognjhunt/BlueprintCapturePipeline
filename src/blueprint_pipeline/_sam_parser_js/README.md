# @meta-sam/parser

`@meta-sam/parser` turns a stream of structural Responses API events into typed,
immutable SAM 3 image or video segmentation snapshots. It parses response output;
it does not create requests, choose models, upload media, or depend on an SDK at
runtime.

## Installation

```sh
npm install @meta-sam/parser
```

## Quick start

Pass an `AsyncIterable<ResponsesEvent>` from your Responses API client or transport
to `parseImageStream` or `parseVideoStream`. The example below feeds the parser
one frame of SAM 3.1 output exactly as the API emits it, split across two response
deltas so that the snapshots demonstrate cumulative state.

<!-- readme-example -->

```ts
import {
  decodeMaskToRaster,
  decodeMaskToRLE,
  decodeMaskToSVGPath,
  frameIndexOf,
  parseVideoStream,
  recordsOfKind,
  type ResponsesEvent,
  type VideoSegmentationResult,
  type VideoSegmentationSnapshot,
} from '@meta-sam/parser';

// Frame 0 of a 320×334 clip, prompt "button", two objects. One line per frame:
// `<Nf>` then comma-separated `id<|box;…|><|mask;…|>` records.
const outputText =
  '<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>' +
  "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>" +
  ',1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>' +
  '<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n';

async function* responseEvents(): AsyncIterable<ResponsesEvent> {
  // Deltas may split anywhere, including inside a mask payload.
  const split = outputText.indexOf(',1<|box');

  yield {
    type: 'response.output_text.delta',
    item_id: 'message-1',
    output_index: 0,
    content_index: 0,
    delta: outputText.slice(0, split),
  };
  yield {
    type: 'response.output_text.delta',
    item_id: 'message-1',
    output_index: 0,
    content_index: 0,
    delta: outputText.slice(split),
  };
  yield {
    type: 'response.output_text.done',
    item_id: 'message-1',
    output_index: 0,
    content_index: 0,
    text: outputText,
  };
  yield { type: 'response.completed' };
}

const parsed = parseVideoStream(responseEvents());

const snapshots: VideoSegmentationSnapshot[] = [];
for await (const snapshot of parsed) {
  snapshots.push(snapshot);
  console.log(snapshot.revision, snapshot.records.length);
}

const result: VideoSegmentationResult = await parsed.finalResult;
if (result.outcome.status !== 'completed') {
  console.warn('Segmentation ended early:', result.outcome);
}
for (const diagnostic of result.diagnostics) {
  console.warn(`${diagnostic.code} on line ${diagnostic.line}: ${diagnostic.message}`);
}

// Two box records and two mask records, object IDs "0" and "1", frame 0.
const [maskRecord] = recordsOfKind(result.records, 'mask');
if (maskRecord !== undefined) {
  const raster = decodeMaskToRaster(maskRecord.mask); // 27 × 60 = 1620 bytes of 0/1
  const cocoRle = decodeMaskToRLE(maskRecord.mask); // size [27, 60]
  const svgPath = decodeMaskToSVGPath(maskRecord.mask);
  console.log(
    frameIndexOf(maskRecord), // 0
    maskRecord.objectId, // "0"
    maskRecord.bounds, // { left: 211, top: 228, right: 271, bottom: 255 }
    raster.length,
    cocoRle.counts,
    svgPath,
  );
}
```

The parser never sends the request itself. With the official OpenAI SDK, the
`AsyncIterable` returned by `client.responses.create({ ..., stream: true })` can be
passed directly; see [Runtime and compatibility](#runtime-and-compatibility).

### Media entry points

`parseImageStream` and `parseVideoStream` are the ergonomic entry points for SAM 3
segmentation. Each one applies the matching segmentation format, so these pairs are
equivalent:

```ts
parseImageStream(events);
parseResponsesStream(events, formats.segmentation.image());

parseVideoStream(events);
parseResponsesStream(events, formats.segmentation.video());
```

Use `parseResponsesStream` with an explicit format to reuse one format across
several streams, or to parse a different output-text format through the same
stream lifecycle.

### SAM 3 text prompts

When the adjacent request uses a SAM 3 text concept prompt, pass a **short noun
phrase** that names what should be segmented, such as `"yellow school bus"`,
`"red car"`, or `"person wearing a red hat"`. SAM 3 concept prompts are not
commands or questions: do not use forms such as `"Segment the person"` or
`"Which object could clean the table?"`. This constraint comes from the
[primary SAM 3 paper](https://arxiv.org/abs/2511.16719v2); it belongs to request
construction rather than parser configuration.

## Snapshots and final results

`parseResponsesStream(source, format)`, and therefore `parseImageStream(source)`
and `parseVideoStream(source)`, returns a
`ParsedResponsesStream<Event, Result>`. It is both an `AsyncIterable` of snapshots
and the owner of a stable `finalResult` promise.

- **Snapshots are cumulative.** Every snapshot contains all accepted records and
  diagnostics so far, not only the latest delta. Arrays, records, masks, frame
  references, diagnostics, snapshots, and final results are frozen.
- **Chunks may split anywhere.** The parser buffers output text across event
  boundaries. A snapshot is emitted only when parsing changes the current view;
  multiple completed lines in one input delta may appear together in one snapshot.
- **`revision` identifies view updates.** It increases only when records or
  diagnostics change.
- **`records` preserve output order.** Text, box, and complete mask records
  share one append-only list. A mask record also has a per-object `identity` and
  `revision`.
- **`rawOutput` is the exact accumulated output text.** Use structured records for
  application behavior and retain `rawOutput` for inspection or logging.
- **`diagnostics` are recoverable format problems.** A malformed structured line is
  omitted from `records` and reported with `severity`, `code`, `message`, `line`,
  and `raw`; parsing continues with later lines.

After normal iteration completes, await `finalResult` for the final cumulative view
and its `outcome`:

| Outcome                                               | Meaning                                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `{status: 'completed'}`                               | A valid `response.completed` event followed one finalized output-text lane.                |
| `{status: 'incomplete', reason: 'response', detail?}` | The API emitted `response.incomplete`; `detail`, when present, is its reason.              |
| `{status: 'incomplete', reason: 'eof'}`               | The event source ended without a terminal response event. Parsed output is still returned. |

Accessing `finalResult` before requesting an iterator drains the source internally
and suppresses intermediate snapshots. This is the simplest mode when only the
final result is needed:

```ts
const result = await parseVideoStream(responseEvents()).finalResult;
```

## One consumer and early exit

A parsed stream has exactly one consumer mode:

- Request its async iterator first to consume snapshots. The same `finalResult`
  promise may be read during or after that iteration.
- Read `finalResult` first to consume the source without snapshots.
- A second iterator, or an iterator requested after final-only consumption starts,
  throws `ResponsesStreamConsumedError`.
- Breaking out of `for await` before completion requests upstream cleanup through
  `return()` when available and rejects `finalResult` with
  `ResponsesStreamAbortedError`.

Do not break after receiving a snapshot if the final result is still required.
Continue iteration to completion, or use final-only consumption.

## Segmentation records

Choose the entry point, or the format, that matches the input media:

```ts
const imageSnapshots = parseImageStream(responseEvents());
const videoSnapshots = parseVideoStream(responseEvents());

const imageFormat = formats.segmentation.image();
const videoFormat = formats.segmentation.video();
```

### SAM API output

SAM 3.1 returns segmentation as special-token text in one `output_text` lane: one
line per frame, with boxes and masks inline.

```text
<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=<frameW>;h=<frameH>|><|mask;x=0;y=0;data=<H>,<W>,<enc>payload|>,id<|box;...|><|mask;...|>
```

- `<Nf>` is the zero-based frame index. Frames with no visible object emit no line,
  so indices can skip; read the explicit value rather than counting lines. A single
  image is emitted as frame `<0f>`.
- Records are comma-separated. Each is a bare integer **object id**, one box, then
  one mask. The id is stable for an object across the frames of one response and is
  not a dense `0`-based sequence: a line may carry `0` and `2`. The parser retains
  it as a string in `objectId` and never derives it from position.
- The box corners `x1`, `y1`, `x2`, `y2` and the frame size `w`, `h` are source
  pixels; `x2` and `y2` are inclusive on the wire. The parser normalizes every box to
  half-open `left`, `top`, `right = x2 + 1`, `bottom = y2 + 1`.
- The mask is `data=H,W,<enc>payload`: the raster's **height then width**, then one
  encoding character — `~` for `lossless` (the API default) or `!` for `one_bit` —
  followed by the complete payload. The parser stores it as
  `mask: { encoding, width, height, payload }` and attaches the box as `bounds`, so
  a renderer knows where the `H × W` raster sits inside the `w × h` frame.
- The payload is **base85, not base64**. After the marker, its digits are the
  printable ASCII characters `!` through `{` minus the wire delimiters
  `" \ , ; < > |`, so it contains `*`, `$`, brackets, and backticks, and a base64
  regex will not match it. Only the first character after `H,W,` is the encoding
  marker; `!` is also digit zero, so payloads routinely start with a run of `!`.
  Treat the payload as opaque and pass it through unchanged.

Every record produces one `SegmentationBoxRecord` followed by one
`SegmentationMaskRecord`, in output order. An empty lane — zero matches — is a
valid completed response with no records. Any non-empty line that does not begin
with `<` is retained as a `SegmentationTextRecord`; a line that begins with `<`
but is not a valid API line produces one `malformed_record` diagnostic and no
records for that line, and earlier lines are unaffected.

For image streams, `<0f>` is required and the normalized records carry no `frame`;
for video streams, records carry `frame: { frameIndex }`. The full grammar is
specified in [`protocol/sam3.md`](https://github.com/meta-models/meta-sam/blob/main/protocol/sam3.md);
the shared conformance corpus includes a captured API response, containing only
protocol output, that exercises this path in both the TypeScript and Python parsers.

### Reading records

`records` is one ordered list of every record kind, so applications usually select
a kind or read a frame index. Two helpers cover both:

```ts
const masks = recordsOfKind(result.records, 'mask');
const frameIndex = frameIndexOf(result.records[0]);
```

`recordsOfKind` returns a frozen list narrowed to the requested kind, such as
`readonly SegmentationMaskRecord[]` for `'mask'`. `frameIndexOf` returns the frame
index of any record, and `undefined` for a text record or an unframed record, so
callers do not test for the `frame` property before reading it.

### Complete mask payloads

Mask records carry one complete `one_bit` or `lossless` base85 payload on a
single logical line. The parser may receive that line in arbitrarily split
response deltas, but it does not emit partial masks. Before adding a
`SegmentationMaskRecord`, it validates the encoding, positive JavaScript-safe
dimensions and area, payload syntax, and complete decoder finalization.
`one_bit` payloads must pass a unique canonical round trip. `lossless` payloads use
a strict packed
envelope, but trailing packed bytes and alternate unused finalization bytes may
encode the same raster and are accepted; no canonical lossless encoder is
exposed.

```ts
const mask = {
  encoding: 'one_bit',
  width: 5,
  height: 5,
  payload: '!!!!!(QO(0lu8?',
} as const;
const raster = decodeMaskToRaster(mask);
const cocoRle = decodeMaskToRLE(mask);
const svgPath = decodeMaskToSVGPath(mask);
```

`decodeMaskToRaster` validates the payload again and returns a row-major
`Uint8Array` of exactly `width * height` entries. Every entry is `0` or `1`.
`decodeMaskToRLE` returns exact COCO compressed RLE with `[height, width]` size;
it transposes the row-major SAM raster into COCO column-major order before
encoding. `decodeMaskToSVGPath` traces that same RLE as polygonal `M`/`L`/`Z`
subpaths and returns `''` for an empty mask. This SVG path intentionally differs
from the smoothed quadratic contour used by `@meta-sam/graphics`.

All three functions perform the same structural mask validation, covering
supported encodings, positive JavaScript-safe dimensions and area, packed
payload shape and alphabet, prefixes, groups, tails, and decoder finalization, but no project-defined area or payload quota ceiling.
Unsupported encodings, invalid dimensions, malformed packing, truncated data,
and invalid decoder finalization throw `InvalidSegmentationMaskError`. JavaScript
allocation failures such as `RangeError` are wrapped in that error with their
original `cause`. For `one_bit`, trailing or noncanonical data also throws;
accepted `lossless` spellings follow the non-unique behavior above.

A mask record's `identity` is stable for its media, frame, and object. Its
`revision` starts at `1` and increases when another complete mask for the same
identity is accepted. Previous mask records remain in the cumulative record list.

## Error handling

All package-specific errors extend `ResponsesStreamError` and expose a stable
`code`. Handle expected categories with `instanceof`, and preserve `cause` when
reporting wrapped parser or source failures.

```ts
import {
  ResponsesStreamError,
  ResponsesStreamRefusalError,
  parseVideoStream,
} from '@meta-sam/parser';

const parsed = parseVideoStream(responseEvents());

try {
  await parsed.finalResult;
} catch (error) {
  if (error instanceof ResponsesStreamRefusalError) {
    console.error('The response was refused:', error.message);
  } else if (error instanceof ResponsesStreamError) {
    console.error(error.code, error.message, error.cause);
  } else {
    throw error;
  }
}
```

| Error                          | `code`                 | When it is used                                                         |
| ------------------------------ | ---------------------- | ----------------------------------------------------------------------- |
| `ResponsesStreamConsumedError` | `stream_consumed`      | A second consumption mode is requested.                                 |
| `ResponsesStreamAbortedError`  | `stream_aborted`       | Snapshot iteration stops before terminal completion.                    |
| `ResponsesStreamFailedError`   | `response_failed`      | A `response.failed` event is received.                                  |
| `ResponsesStreamEventError`    | `response_error`       | A stream `error` event is received.                                     |
| `ResponsesStreamLaneError`     | `response_lane`        | Output-text lane identity, ordering, or finalized text is inconsistent. |
| `ResponsesStreamRefusalError`  | `response_refusal`     | A refusal delta or completion is received.                              |
| `ResponsesStreamParserError`   | `parser_error`         | A non-package error escapes a response-format parser.                   |
| `ResponsesStreamSourceError`   | `source_error`         | Creating, reading, or closing the source iterator fails.                |
| `InvalidSegmentationMaskError` | `invalid_mask_payload` | A complete mask cannot be strictly decoded.                             |

Terminal stream failures reject both iteration and `finalResult`. On normal EOF,
an explicit terminal response, early exit, or failure, the parser closes the
upstream owner exactly once through `aclose()` or `close()` when present, otherwise
through a created iterator's `return()`. A source owner can therefore release
transport resources even when no iterator was started.

## Public API

Only the package root is public; deep imports are not supported.

### Runtime exports

| Export                         | Purpose                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `parseImageStream`             | Parse one event stream as image segmentation.                                                         |
| `parseVideoStream`             | Parse one event stream as video segmentation.                                                         |
| `parseResponsesStream`         | Lazily consume one structural Responses event stream with a response format.                          |
| `formats`                      | Frozen registry containing zero-argument `segmentation.image()` and `segmentation.video()` factories. |
| `recordsOfKind`                | Select one record kind from a record list as a frozen narrowed list.                                  |
| `frameIndexOf`                 | Read the frame index of any record, or `undefined` when it has none.                                  |
| `decodeMaskToRaster`           | Strictly decode one complete validated mask to a binary row-major `Uint8Array`.                       |
| `decodeMaskToRLE`              | Convert one complete mask to exact COCO compressed RLE.                                               |
| `decodeMaskToSVGPath`          | Convert one complete mask to an exact polygonal SVG path.                                             |
| `ResponsesStreamError`         | Base class for package-specific errors.                                                               |
| `ResponsesStreamConsumedError` | Invalid repeated or conflicting consumption.                                                          |
| `ResponsesStreamAbortedError`  | Early snapshot-iteration termination.                                                                 |
| `ResponsesStreamFailedError`   | API `response.failed` event.                                                                          |
| `ResponsesStreamEventError`    | API stream `error` event.                                                                             |
| `ResponsesStreamLaneError`     | Invalid or interleaved output-text lanes.                                                             |
| `ResponsesStreamRefusalError`  | API refusal event.                                                                                    |
| `ResponsesStreamParserError`   | Wrapped response-format parser failure.                                                               |
| `ResponsesStreamSourceError`   | Wrapped source iterator failure.                                                                      |
| `InvalidSegmentationMaskError` | Invalid complete mask payload.                                                                        |

### Type exports

- Stream protocol: `ResponsesEvent`, `OutputTextLane`, `ResponseStreamOutcome`,
  `ParsedResponsesStream`, `ResponseFormat`, and `ResponseFormatParser`.
- Segmentation views: `SegmentationSnapshot`, `ImageSegmentationSnapshot`,
  `VideoSegmentationSnapshot`, `SegmentationResult`, `ImageSegmentationResult`,
  and `VideoSegmentationResult`.
- Segmentation records: `SegmentationRecord`, `SegmentationTextRecord`,
  `SegmentationBoxRecord`, `SegmentationMaskRecord`, `SegmentationMask`,
  `SegmentationMaskEncoding`, `SegmentationMaskBounds`, `RLEObject`,
  `FrameReference`, and `SegmentationDiagnostic`.
- Record selection: `SegmentationRecordKind` and
  `SegmentationRecordOfKind<Kind>`, the narrowed record type `recordsOfKind`
  returns.

`ResponseFormat` and `ResponseFormatParser` are public extension points for parsing
other output-text formats with the same stream lifecycle. A parser receives ordered
text chunks through `push`, then receives the terminal `ResponseStreamOutcome`
through `finish`.

## Runtime and compatibility

- ESM only. Use `import`; there is no CommonJS export.
- Supported Node.js versions are `^20.17.0 || >=22.9.0`.
- The package targets ES2022 and has no runtime dependencies.
- The source is structural: `parseResponsesStream` accepts any `AsyncIterable` of
  objects with a string `type` discriminator and reads supported Responses API
  fields when their event type is encountered. Unrelated event types are ignored.
- OpenAI `ResponseStreamEvent` values satisfy the stream's structural input
  requirement; the OpenAI SDK is a development-only compatibility check, not a
  runtime dependency.
- Exactly one finalized `response.output_text` lane is required before
  `response.completed` or `response.incomplete`. The lane is finalized by
  `response.output_text.done` or by `response.content_part.done` for an
  `output_text` part; the live SAM API emits only the latter. Both may arrive
  for one lane, as the OpenAI Responses API emits, when their text is identical.
  Interleaved lanes and finalized text that conflicts with accumulated deltas
  are rejected.

## Related packages

| Package              | Role                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| `@meta-sam/parser`   | Parse structural response events into segmentation snapshots and results. |
| `@meta-sam/graphics` | Retain mask paths and render Canvas 2D overlays from parser records.      |
| `@meta-sam/video`    | Decode media into a Canvas with packet-exact frame metadata and audio.    |
| `@meta-sam/react`    | Provide React bindings over the video and graphics packages.              |

## License

The source is licensed under the SAM License. See `LICENSE` in this package or the
repository root for the license text.
