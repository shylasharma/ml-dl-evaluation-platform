"""
Turns internal exceptions into clear, user-facing messages instead of raw
Python tracebacks. Routers should catch broad exceptions at the boundary and
pass them through `friendly_message`.
"""


import traceback


def exception_location(exc: BaseException) -> str:
    """Short call path (deepest 3 frames) of an exception, for diagnostics."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return ""
    # Skip numpy's own internals so the app/library call site (e.g. imblearn) is what shows up.
    frames = [f for f in frames if "/numpy/" not in f.filename.replace("\\", "/")] or frames
    parts = []
    for fr in reversed(frames[-3:]):
        short = "/".join(fr.filename.replace("\\", "/").split("/")[-2:])
        parts.append(f"{short}:{fr.lineno} {fr.name}")
    return " <- ".join(parts)


class FriendlyError(Exception):
    """An error whose message is already safe to show to the end user."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def friendly_message(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, FriendlyError):
        return exc.message
    if "could not be trained" in text or "training failed" in text or "Could not apply" in text:
        return text
    if isinstance(exc, (ValueError, KeyError)):
        return f"The request could not be processed: {text}"
    if isinstance(exc, MemoryError):
        return "The operation ran out of memory. Try a smaller dataset, fewer models, or disable resampling."
    # Fallback: never leak raw tracebacks, but do name the error type so it can be diagnosed
    detail = " ".join(text.split())[:200]
    where = exception_location(exc)
    where = f" [at {where}]" if where else ""
    return (f"An unexpected error occurred ({type(exc).__name__}: {detail}){where}. "
            f"Please check your configuration and try again.")