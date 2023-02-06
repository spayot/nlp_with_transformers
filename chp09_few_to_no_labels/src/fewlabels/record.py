import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EvalFcn(Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        ...

@dataclass
class Recorder:
    metrics: set[str]

    def __init__(self, metrics_names: list[str]):
        self.records = {metric: defaultdict(list) for metric in metrics_names}
        self.metrics = set(metrics_names)

    def add_record(self, record: dict[str, float], name: str) -> None:
        assert self.metrics.issubset(record.keys()), f"record {record} is missing some metrics."
        for metric in self.metrics:
            self.records[metric][name].append(record.get(metric))

    def to_pickle(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.records, f)

    @classmethod
    def from_pickle(cls, path: str):
        with open(path, "rb") as f:
            records = pickle.load(f)
            recorder = cls(metrics_names=records.keys())
            recorder.records = records
            return recorder


    