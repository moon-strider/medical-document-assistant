# Backend contracts

PostgreSQL is the authority for uploaded sources, extracted evidence, and saved runs. A source becomes searchable only when its evidence and `ready` state are published together. An answer is saved only while its run is active and the collection has not changed since the run began. This keeps the answer tied to the evidence available for that run.

Deleting a source removes it from future search, but cleanup of its private file may finish later. Deletion cannot retract text already sent to an inference provider. Search metadata such as document titles cannot support citations; citations must resolve to extracted source text.

See [providers](src/medical_assistant/providers/README.md), [prompts](../docs/prompts.md), and [trace delivery](src/medical_assistant/TELEMETRY.md) for adjacent interfaces.
