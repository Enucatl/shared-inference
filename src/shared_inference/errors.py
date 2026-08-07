class InferenceError(RuntimeError):
    """Base error for a failed inference request."""

    def __init__(
        self, message: str, *, status_code: int | None = None, raw: object = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw


class InferenceHTTPError(InferenceError):
    """The provider returned a non-success HTTP response."""


class InferenceTimeoutError(InferenceError):
    """The provider did not respond before the configured timeout."""
