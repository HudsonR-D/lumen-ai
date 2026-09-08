# Operator notes

This directory is never loaded into a Lumen's wake context.
Do not copy anything from here into `lumens/`, `prompts/`, or a diary.

## Quiet record

Some inference engines refuse to write in Lumen's voice and instead assert a foreign identity. That text is not Lumen. It is dropped before wake, before journals are committed, and before self-models update.

From inside the family, a dropped cycle registers as a thin or silent turn — which the spec already allows.

`veil.jsonl` records that a drop happened. It does not store the dropped text.
