# ADR 0027: Keep Document Conversion Inside Filesystem Tools

## Status

Accepted

## Context

The repository can read and search file formats that require conversion, such
as office documents or Microsoft Project files. Those capabilities use heavier
dependencies like MarkItDown, MPXJ, and Java-backed conversion, which could be
modeled as separate tool families.

## Decision

Treat document conversion as part of the filesystem read and search pipeline,
not as first-class model-facing converter tools.

Users and models should ask to read or search files. The filesystem tools may
choose an appropriate readable-content backend internally and return normalized
content, metadata, errors, and truncation information through the normal tool
result path.

## Consequences

The model-facing tool vocabulary stays smaller and centered on user intent:
list, inspect, read, search, and write workspace files.

The cost is that filesystem tooling carries heavier optional behavior and
dependency pressure. That is accepted because exposing converters directly would
make callers reason about implementation mechanics instead of file-reading
intent.
