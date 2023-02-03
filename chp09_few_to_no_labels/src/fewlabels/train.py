import json
import logging

import datasets
import numpy as np
import transformers as tfm
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report

from .eval import SlicedTrainingEvaluator


# opportunity to refactor: move the default training args into a json file
def load_training_args(config_path: str, output_dir: str):
    with open(config_path, "r") as f:
        config: dict = json.load(f)
    config["output_dir"] = output_dir

    return tfm.TrainingArguments(**config)


class SlicedTrainingRunner:
    def __init__(
        self,
        model_ckpt: str,
        dataset: datasets.Dataset,
        num_labels: int,
        output_dir: str,
        path_to_training_config: str = None,
    ) -> None:
        self.model_ckpt = model_ckpt
        self.dataset = dataset
        self.training_args = load_training_args(path_to_training_config, output_dir)

        logging.info(f"loading `{model_ckpt}` config")
        self._load_model_config(model_ckpt, num_labels)

        logging.info(f"loading `{model_ckpt}` tokenizer")
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(model_ckpt)

    def _load_model_config(self, model_ckpt: str, num_labels: int) -> tfm.AutoConfig:
        self.config = tfm.AutoConfig.from_pretrained(model_ckpt)
        self.config.num_labels = num_labels
        self.config.problem_type = "multi_label_classification"

    def evaluate_tfm_model_on_all_slices(
        self,
        strategy: str,
        evaluator: SlicedTrainingEvaluator,
        train_slices: list[list[int]],
    ) -> SlicedTrainingEvaluator:
        """trains classifier on different slices."""

        for train_slice in train_slices:
            trainer = self.train_on_single_slice(train_slice=train_slice)
            y_true, y_pred = self.get_trainer_preds_on_test_set(trainer)
            evaluator.add_f1_scores(y_true, y_pred, strategy)

        return evaluator

    def train_on_single_slice(self, train_slice: np.ndarray) -> tfm.Trainer:

        model = tfm.AutoModelForSequenceClassification.from_pretrained(
            self.model_ckpt, config=self.config
        )

        trainer = tfm.Trainer(
            model=model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.dataset["train"].select(train_slice),
            eval_dataset=self.dataset["valid"],
        )

        trainer.train()

        return trainer

    def get_trainer_preds_on_test_set(
        self,
        trainer: tfm.Trainer,
    ) -> tuple:

        pred = trainer.predict(self.dataset["test"])
        y_pred = sigmoid(pred.predictions)  # from logits to sigmoid
        y_pred = (y_pred > 0.5).astype(float)
        return pred.label_ids, y_pred


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)  # from logits to sigmoid
    y_pred = (y_pred > 0.5).astype(float)

    clf_report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True
    )

    return {
        "micro f1": clf_report["micro avg"]["f1-score"],
        "macro f1": clf_report["macro avg"]["f1-score"],
    }
