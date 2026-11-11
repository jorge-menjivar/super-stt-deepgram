# Super Voice — Deepgram backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-deepgram/coverage.json)](https://super-libre.github.io/super-voice-deepgram/)

[Deepgram](https://developers.deepgram.com/docs/models-languages-overview)'s
hosted speech-to-text, `nova-3`, as a backend for
[Super Voice](https://github.com/jorge-menjivar/super-voice). The audio goes to
Deepgram, so it needs an API key and a connection rather than a GPU. It hears
your vocabulary: the terms go to the model as
[keyterms](https://developers.deepgram.com/docs/keyterm) with every recording.
Deepgram takes terms, not instructions, so your prompt is not sent.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends and drives each one over a `/v1` HTTP contract. This is one such
backend: a sandboxed WebAssembly component that answers `POST /v1/transcribe`,
with no network access but `api.deepgram.com`.

## License

GPL-3.0-only. See `LICENSE`.
