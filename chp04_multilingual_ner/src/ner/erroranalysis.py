import datasets
import numpy as np
import pandas as pd
import torch
import transformers as tfm

from .preproc import IGNORED_TAG_ID


class TokensLossDataFrameBuilder:
    def __init__(
        self,
        trainer: tfm.Trainer,
        data_collator: tfm.DataCollatorForTokenClassification,
    ):
        """
        - calculate token-level loss on dataset
        - convert to sentence level dataframe
        - explode to token level dataframe
        - analyze loss grouped by
        """
        self.trainer = trainer
        self.data_collator = data_collator
        self._tokens_df: pd.DataFrame = None

        self.id2label_with_ignore = trainer.model.config.id2label.copy()
        self.id2label_with_ignore[IGNORED_TAG_ID] = "IGN"

    def build_dataframe(self, dataset: datasets.Dataset):
        ds = self._calculate_token_level_loss_on_dataset(dataset)
        df = self._convert_dataset_to_pandas(ds)
        return self._convert_to_token_level_dataframe(df)

    def _calculate_token_level_loss_on_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:

        return dataset.map(
            calculate_loss_and_predictions_on_batch,
            fn_kwargs={"trainer": self.trainer, "data_collator": self.data_collator},
            batched=True,
            batch_size=32,
        )

    def _convert_dataset_to_pandas(self, dataset: datasets.Dataset) -> pd.DataFrame:
        self._assert_dataset_is_valid(dataset)

        df = dataset.to_pandas()

        # convert to tokens
        df["input_tokens"] = df.input_ids.map(
            self.trainer.tokenizer.convert_ids_to_tokens
        )

        # convert tags to readable format
        df["predicted_labels"] = df["predicted_labels"].apply(
            self._convert_tag_ids_to_str
        )

        # remove padding
        df["predicted_labels"] = df.apply(
            lambda row: self._remove_padding(
                row["predicted_labels"], target_length=len(row["input_ids"])
            ),
            axis=1,
        )

        # convert tags to readable format
        df["labels"] = df["labels"].apply(self._convert_tag_ids_to_str)

        # remove padding
        df["loss"] = df.apply(
            lambda row: self._remove_padding(
                row["loss"], target_length=len(row["input_ids"])
            ),
            axis=1,
        )
        return df

    def _assert_dataset_is_valid(self, dataset):
        EXPECTED_FEATURES = ["input_ids", "predicted_labels", "labels", "loss"]
        dataset_features = dataset.features.keys()
        for feature in EXPECTED_FEATURES:
            assert (
                feature in dataset_features
            ), f"feature {feature} is missing from the input dataset. dataset features: {dataset_features}"

    def _convert_to_token_level_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_tokens = df.apply(pd.Series.explode)
        df_tokens = df_tokens.query("labels != 'IGN'")
        df_tokens["loss"] = df_tokens["loss"].astype(float).round(2)

        return df_tokens

    def _convert_tag_ids_to_str(self, ids: list[int]) -> list[str]:
        return [self.id2label_with_ignore[id] for id in ids]

    @staticmethod
    def _remove_padding(seq: list, target_length: int):
        return seq[:target_length]


def calculate_loss_and_predictions_on_batch(
    batch: dict[str, torch.tensor],
    trainer: tfm.Trainer,
    data_collator: tfm.DataCollatorForTokenClassification,
) -> dict[str, np.ndarray]:
    "computes predicted labels and loss between actual labels and predicted, given a batch of encoded examples."

    # turn batch into a list of dictionaries for individual examples
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]

    # does padding on inputs AND labels
    batch = data_collator(features)

    return calculate_loss(batch, trainer)


def calculate_loss(
    batch: dict[str, list],
    trainer: tfm.Trainer,
) -> dict[str, np.ndarray]:

    with torch.no_grad():
        outputs = trainer.model(
            batch["input_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )

    n_classes = len(trainer.model.config.id2label)

    loss = torch.nn.functional.cross_entropy(
        input=outputs.logits.view(-1, n_classes),
        target=batch["labels"].view(-1),
        reduction="none",
    )

    batch_size = batch["labels"].size()[0]

    return {
        "loss": loss.view(batch_size, -1).numpy(),
        "predicted_labels": outputs.logits.argmax(-1).numpy(),
    }


class NERLossAnalyzer:
    def __init__(self, df_tokens: pd.DataFrame):
        self.df_tokens = df_tokens

    def loss_groupby_analysis(
        self, groupby: str, sort: str = "mean", ascending: bool = False, head: int = 10
    ) -> pd.DataFrame:
        return (
            self.df_tokens.groupby(groupby)[["loss"]]
            .agg(["count", "mean", "sum"])
            .droplevel(level=0, axis=1)
            .sort_values(sort, ascending=ascending)
            .reset_index()
            .round(2)
            .head(head)
            .T
        )
