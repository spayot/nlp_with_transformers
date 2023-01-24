from collections import defaultdict

import datasets
import numpy as np
import transformers as tfm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


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
    ) -> None:
        for train_slice in self.train_slices:
            ds_train_sample = self.ds["train"].select(train_slice)
            y_train = np.array(ds_train_sample["label_ids"])
            y_test = np.array(self.ds["test"]["label_ids"])

            # fit pipeline
            _ = pipe.fit(ds_train_sample["text"], y_train)

            # generate preds and evaluate
            y_pred_test = pipe.predict(self.ds["test"]["text"])
            clf_report = classification_report(
                y_test,
                y_pred_test,
                target_names=self.mlb.classes_,
                zero_division=0,
                output_dict=True,
            )

            self.f1_scores["macro"][strategy].append(
                clf_report["macro avg"]["f1-score"]
            )
            self.f1_scores["micro"][strategy].append(
                clf_report["micro avg"]["f1-score"]
            )
