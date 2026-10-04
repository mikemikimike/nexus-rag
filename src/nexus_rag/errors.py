"""How an error reads in a report."""


def describe(error: BaseException) -> str:
    """The error's message, or its type name when the message is empty."""
    return str(error) or type(error).__name__
