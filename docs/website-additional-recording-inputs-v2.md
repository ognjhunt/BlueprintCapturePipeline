# Same-job recording inputs

The normal customer path is upload, Blueprint reconstruction and evaluation,
then a reconstruction/result/proposal when the corresponding work completes.
Atlas owns source-frame recovery and bounded scene completion. An earlier
coverage hint must not send the customer back to film or confirm the draft.
Unknown source regions and generated estimates remain recorded internally.

An owner may optionally add a recording. WebApp issues `append_views`; the
signed link declares `recording_scope: additional_views`. The historical
`additional_views` request continues to mean a complete walkthrough. Ordinary
complete recordings retain the byte-compatible v1 continuation.

`website_capture_continuation.v2` carries an exact ordered active recording set,
stopping at the latest complete replacement. Every original remains immutable.
The shared vector is `tests/fixtures/website-additional-views-vector.json` and
must match WebApp's vector byte for byte. Source generations, size, CRC32C,
available SHA-256, original bundle identity and same-job lineage are verified.

The existing Atlas worker and direct-preparation consumer resolve these sources
into a cached, audio-free method video. Decoded RGB pixels, native presentation
cadence, rotation and each clip's identity are retained. Different image sizes
are center padded; quality and overlap tests exclude only that known padding.
The carrier's final-frame display hold is explicit, not measured task duration.
The derivative never replaces a raw manifest URI or claims common camera
coordinates, geometric registration, reuse of a changed scene reconstruction,
metric accuracy, simulator readiness or physical truth.

The grant binds all original identities and the derived method digest. Existing
permissions, provider terms, spend receipts, caps, holds and current-source
checks still apply. Reopening unchanged inputs reuses the retained derivative;
changing observations may require reconstruction. No new queue or provider is
introduced. Deploy this consumer before the v2 WebApp producer. Optional clips
must not delay release or make another customer step compulsory.

Validation: 193 focused source, media, ingestion, preview, preparation, survey
and resume tests passed, including real FFmpeg video in both clip orders,
pixel/timestamp preservation, rotation, audio removal, padding-aware analysis
and interrupted-cache recovery. Changed-file Ruff passed. These are local
checks; no provider call, live capture, spending or deployment was performed.
