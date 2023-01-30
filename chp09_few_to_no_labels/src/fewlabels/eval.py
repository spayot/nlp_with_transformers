from collections import defaultdict

import datasets
import numpy as np
import transformers as tfm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from .augment import AugmentBatchFcn


class SlicedTrainingEvaluator:
    def __init__(
        self,
        ds: datasets.DatasetDict,
        train_slices: list[np.ndarray],
        mlb: MultiLabelBinarizer,
    ):
        self.ds = ds
        self.train_slices = train_slices
        self.f1_scores = {"micro": defaultdict(list), "macro": defaultdict(list)}
        self.mlb = mlb

    def evaluate_pipe_on_slices(
        self,
        pipe: Pipeline,
        strategy: str,
        augment_fcn: AugmentBatchFcn = None,
    ) -> None:

        for train_slice in self.train_slices:
            self._evaluate_pipe_on_single_slice(
                pipe,
                strategy,
                train_slice,
                augment_fcn,
            )

    def _evaluate_pipe_on_single_slice(
        self,
        pipe: Pipeline,
        strategy: str,
        train_slice: list[int],
        augment_fcn: AugmentBatchFcn = None,
    ) -> None:

        ds_train_sample = self._generate_training_dataset(
            train_slice,
            augment_fcn,
        )

        y_train = np.array(ds_train_sample["label_ids"])
        y_test = np.array(self.ds["test"]["label_ids"])

        # fit pipeline
        _ = pipe.fit(ds_train_sample["text"], y_train)

        # generate preds and evaluate
        y_pred_test = pipe.predict(self.ds["test"]["text"])

        self.add_f1_scores(y_test, y_pred_test, strategy)

    def _generate_training_dataset(
        self, train_slice: list[int], augment_fcn: AugmentBatchFcn
    ) -> datasets.Dataset:
        ds_train_sample = self.ds["train"].select(train_slice)
        if augment_fcn:
            ds_train_sample = ds_train_sample.map(
                augment_fcn,
                batched=True,
                remove_columns=ds_train_sample.column_names,
            ).shuffle()
        return ds_train_sample

    def add_f1_scores(self, y_test: np.ndarray, y_pred: np.ndarray, strategy: str):
        scores = get_f1_scores(y_test, y_pred, self.mlb.classes_)

        for metric in scores.keys():
            self.f1_scores[metric][strategy].append(scores[metric])


def get_f1_scores(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] = None
) -> dict[str, float]:
    clf_report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True
    )

    return {
        "micro": clf_report["micro avg"]["f1-score"],
        "macro": clf_report["macro avg"]["f1-score"],
    }
