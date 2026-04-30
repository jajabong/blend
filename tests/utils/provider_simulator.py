
import time


class ProviderSimulator:
    """Utility to inject faults, latency, and failures into LLM provider calls."""

    def __init__(self):
        self.latency = 0.0
        self.should_fail = False
        self.exception = Exception("Provider failed")

    def set_latency(self, seconds: float):
        self.latency = seconds

    def set_failure(self, exception: Exception = None):
        self.should_fail = True
        if exception:
            self.exception = exception

    def simulate(self, func, *args, **kwargs):
        """Wraps a provider call to inject faults."""
        if self.latency > 0:
            time.sleep(self.latency)
        if self.should_fail:
            raise self.exception
        return func(*args, **kwargs)

def get_simulator_fixture():
    return ProviderSimulator()
