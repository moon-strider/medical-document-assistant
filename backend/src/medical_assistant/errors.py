class ConflictError(ValueError):
    """Reject a request that conflicts with persisted workflow state.

    Raised when an upload/run idempotency key is reused with different compared
    content, or a conversation already has another queued/running run. The API
    maps this domain error to a conflict response rather than generic invalid
    input. Matching request replays return the stored record instead.
    """

    pass


class TransientIngestError(ValueError):
    """Identify ingestion failures eligible for the worker's bounded retry policy.

    Parser startup/timeouts, unavailable source bytes and invalid parser output
    use this marker; it does not promise that another attempt will succeed.
    The worker converts it into a job error, then storage checks lease ownership
    and the attempt limit before scheduling a retry or failing the source.
    """

    pass
