import os
from time import perf_counter

import datasets
import psutil
from tqdm import tqdm

N_INFERENCES = 100
BATCH_SIZE = 16


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


class TestProcedure:
    def __init__(self):
        self._preproc_perf = PerfCheck()
        self._query_perf = PerfCheck()

    def start_preprocessing(self):
        print("starting transformation:")
        self._preproc_perf.start()

    def stop_preprocessing(self):
        self._preproc_perf.stop()
        print(self._preproc_perf)

    def test_query_performance(
        self,
        dataset: datasets.Dataset,
        field_to_extract: str = "pixel_values",
        batch_size: int = BATCH_SIZE,
        n_inferences: int = N_INFERENCES,
    ):
        print(f"\nquerying {n_inferences} batches of size {batch_size} the dataset")
        self._query_perf.start()
        for i in tqdm(range(n_inferences)):
            _ = dataset[i : i + batch_size][field_to_extract]
        self._query_perf.stop()
        print(self._query_perf)

    def get_record(self):
        preproc_record = self._perf_check_to_dict(self._preproc_perf, "preproc_")
        query_record = self._perf_check_to_dict(self._query_perf, "query_")
        preproc_record.update(query_record)
        return preproc_record

    def _perf_check_to_dict(self, pc: PerfCheck, prefix: str) -> dict:
        return {prefix + k: v for k, v in pc.__dict__.items() if k != "status"}
