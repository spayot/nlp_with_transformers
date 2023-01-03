import os
from time import perf_counter

import psutil


def get_time_and_mem() -> tuple[float, float]:

    return perf_counter(), psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class PerfCheck:
    def __init__(self):
        """measures the impact on RAM and time of a python execution.
        Example:
        >>> pc = PerfCheck()
        >>> pc.start()
        >>> time.sleep(1)
        >>> pc.stop()
        """
        self.memory: tuple[float, float] = None
        self.cpu_perf: tuple[float, float] = None
        self.status: str = "INITIALIZED"

    def start(self) -> None:
        self.memory = self._get_ram_mb()
        self.cpu_perf = self._get_time()
        self.status = "RUNNING"

    def stop(self) -> None:
        self.memory = self._get_ram_mb() - self.memory
        self.cpu_perf = self._get_time() - self.cpu_perf
        self.status = "COMPLETED"

    def print(self) -> None:
        print(f"memory used: {self.after[1] - self.before[1]:.2f}MB")
        print(f"latency: {self.after[0] - self.before[0]:.2f} sec")

    def reset(self) -> None:
        self.__init__()

    def __repr__(self) -> str:
        if self.status == "COMPLETED":
            return f"PerfCheck(status={self.status}, cpu_perf={self.cpu_perf:.2f} sec, memory={self.memory:.2f}MB)"
        return f"PerfCheck(status={self.status})"

    def _get_time(self) -> float:
        return perf_counter()

    def _get_ram_mb(self) -> float:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
