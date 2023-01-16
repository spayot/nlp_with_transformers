from typing import Tuple

import numpy as np
import transformers as tfm
from seqeval import metrics

from .preproc import IGNORED_TAG_ID


class SeqevalConverter:
    def __init__(self, id2label: dict[int, str]):
        """converts"""
        self.id2label = id2label

    def align_predictions(
        self, predictions: np.ndarray, label_ids: np.ndarray
    ) -> Tuple[list[list[str]], list[list[str]]]:
        """Converts raw logits predictions and label_ids into lists of text labels,
        ie the format understood by seqeval.metrics"""
        preds = np.argmax(predictions, -1)
        y_true, y_pred = [], []
        for example, labels in zip(preds, label_ids):
            y_true.append(self._convert_labels(labels))
            y_pred.append(self._convert_preds(example, labels))

        return y_true, y_pred

    def _is_valid(self, label: int) -> bool:
        return label != IGNORED_TAG_ID

    def _convert_labels(self, labels: list[int]):
        return [self.id2label[label] for label in labels if self._is_valid(label)]

    def _convert_preds(self, example: list[int], labels: list[int]):
        return [
            self.id2label[pred]
            for pred, label in zip(example, labels)
            if self._is_valid(label)
        ]

    def compute_metrics(self, eval_pred: tfm.EvalPrediction) -> dict[str, float]:
        y_true, y_preds = self.align_predictions(
            predictions=eval_pred.predictions,
            label_ids=eval_pred.label_ids,
        )
        return {"f1": metrics.f1_score(y_true, y_preds)}
