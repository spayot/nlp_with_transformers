# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import pickle
from pathlib import Path
from time import perf_counter

import evaluate
import numpy as np
import torch

accuracy_score = evaluate.load("accuracy")

QUERY = "i would like to make a trip from Paris to Roma"


class PerformanceBenchmark:
    def __init__(
        self,
        pipeline,
        dataset,
        optim_type: str = "baseline Bert",
        class_label: str = "intent",
        batch_size: int = 48,
    ) -> None:
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type
        self.class_label = dataset.features[class_label]

    def compute_accuracy(self, batch_size: int = 48) -> float:
        n_batches = len(self.dataset) // batch_size + 1

        preds, labels = [], []
        for batch_idx in range(n_batches):
            batch = self.dataset[batch_size * batch_idx : batch_size * (batch_idx + 1)]
            pred = self.pipeline(batch["text"])
            preds += [
                self.class_label.str2int(label)
                for label in map(lambda x: x["label"], pred)
            ]
            labels += batch["intent"]

        acc = accuracy_score.compute(predictions=preds, references=labels)

        print(f"Model Accuracy:   {acc['accuracy']:>10.2%}")

        return acc

    def compute_size(self) -> dict:
        # save model in temporary path
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("tmp_model.pt")
        torch.save(state_dict, tmp_path)

        # calculate size
        size_mb = tmp_path.stat().st_size / 1024**2
        # delete file
        tmp_path.unlink()

        print(f"Model Size (MB): {size_mb:>10.1f}")
        return {"size": size_mb}

    def compute_latency(self) -> dict:
        # warmup
        for _ in range(10):
            _ = self.pipeline(QUERY)

        latencies = []
        # calculate latency 100 times
        for _ in range(100):
            start = perf_counter()
            _ = self.pipeline(QUERY)
            latencies.append(perf_counter() - start)

        latencies = np.array(latencies) * 1000

        print(f"Avg. Latency (ms): {latencies.mean():.2f} +/- {latencies.std():.2f}")

        return {
            "latency_ms_avg": latencies.mean(),
            "latency_ms_std": latencies.std(),
        }

    def run_benchmark(self, batch_size: int = 48) -> dict[str, dict]:
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.compute_latency())
        metrics[self.optim_type].update(self.compute_accuracy(batch_size))

        return metrics


class OnnxPerformanceBenchmark(PerformanceBenchmark):
    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def compute_size(self) -> dict:
        size_mb = Path(self.model_path).stat().st_size // 1024**2
        print(f"Model Size (MB): {size_mb:>10.1f}")
        return {"size": size_mb}

    def compute_accuracy(self, batch_size: int = 48) -> dict:
        """Note: batch_sizse is not used in this code, but kept to respect
        the same protocol for compute_accuracy than the parent class.
        """

        preds, labels = [], []
        for batch in self.dataset:
            pred = self.pipeline(batch["text"])
            pred = self.class_label.str2int(pred[0]["label"])
            preds.append(pred)

            labels.append(batch["intent"])

        acc = accuracy_score.compute(predictions=preds, references=labels)

        print(f"Model Accuracy:   {acc['accuracy']:>10.2%}")

        return acc


class PerfMetrics:
    def __init__(self, path: str):
        self.metrics: dict[str, dict] = {}
        self.path = path

    def to_pkl(self, path: str = None) -> None:
        if not path:
            path = self.path
        with open(path, "wb") as f:
            pickle.dump(self.metrics, f)

    @classmethod
    def from_pkl(cls, path: str):
        with open(path, "rb") as f:
            metrics = pickle.load(f)

        pm = cls(path=path)
        pm.metrics = metrics
        return pm

    def update(self, new_metrics: dict) -> None:
        self.metrics.update(new_metrics)

    def __repr__(self) -> str:
        return "\n".join(f"{k:<25}{v}" for k, v in self.metrics.items())

    def __getitem__(self, key) -> dict:
        return self.metrics[key]
