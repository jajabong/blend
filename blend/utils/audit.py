
import json
import time
from functools import wraps

from blend.core.executor import L3Output


def audit_request(func):
    """Decorator to audit Token consumption for Blend API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Log to a simple file
        if isinstance(result, L3Output):
            audit_entry = {
                "timestamp": time.time(),
                "model": result.model_used,
                "tokens": result.tokens_used,
                "duration": duration,
                "efficiency": result.tokens_used / duration if duration > 0 else 0
            }
            with open("audit_log.jsonl", "a") as f:
                f.write(json.dumps(audit_entry) + "\n")

        return result
    return wrapper
