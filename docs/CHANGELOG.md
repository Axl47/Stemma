# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `audio-embed classify --lyrics <vectors.jsonl>` adds a per-song lyric vector (e.g. the mean of its passage embeddings) as extra features, weighted by `--lyrics-weight` (default 2). Songs without one are scored from sound alone. On held-out artists it raised recall@3 by about 1.2 points and auto-check precision from 77% to 81%. Without the flag, output is unchanged.
- `audio-embed languages` guesses each song's language from its lyrics, and `classify` uses an optional `language` field to place songs that sound alike but are sung in different languages.
