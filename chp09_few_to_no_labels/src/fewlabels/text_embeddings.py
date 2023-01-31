from typing import Any

import datasets
import numpy as np
import torch
import transformers as tfm

from . import eval


class TransformerWithMeanPooling(torch.nn.Module):
    def __init__(self, model: tfm.AutoModel):
        """turns full text into a single embedding, using mean pooling
        on the last hidden state layer."""
        super().__init__()
        self.model = model

    def forward(self, **inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        model_output = self.model(**inputs)
        return self.mean_pooling(
            model_output.last_hidden_state, inputs["attention_mask"]
        )

    def mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = self._expand_mask_over_emb_dim(
            attention_mask, dim=last_hidden_state.size()
        ).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def _expand_mask_over_emb_dim(
        self, attention_mask: torch.Tensor, dim: int
    ) -> torch.Tensor:
        return attention_mask.unsqueeze(-1).expand(dim)


class KNNTagger:
    def __init__(
        self,
        k: int,
        threshold: int,
    ):
        self.train_ds: datasets.Dataset = None
        self.k = k
        self.threshold = threshold
        assert (
            threshold <= k
        ), f"threshold {threshold} should be lower or equal to k ({k})"

    def fit(self, train_ds: datasets.Dataset) -> None:
        self.train_ds = train_ds

    def predict(self, ds: datasets.Dataset) -> datasets.Dataset:
        assert (
            self.train_ds is not None
        ), "you need to fit the KNNTagger on a training dataset first."
        return ds.map(self.predict_single_example)

    def _extract_tags_from_nearest_neighbors(self, example):
        neighbors = self.train_ds.get_nearest_examples(
            "embeddings",
            query=np.array(example["embeddings"], dtype=np.float32),
            k=int(self.k),
        )
        return neighbors.examples["label_ids"]

    def _select_tags_from_neighbor_tags(
        self, neighbor_labels: list[list[int]]
    ) -> list[int]:
        return list((np.array(neighbor_labels).sum(axis=0) >= self.threshold) * 1)

    def predict_single_example(
        self, example: dict[str, list[Any]]
    ) -> dict[str, list[int]]:
        neighbor_labels = self._extract_tags_from_nearest_neighbors(example)
        return {
            "predicted_labels": self._select_tags_from_neighbor_tags(neighbor_labels)
        }

    def score(
        self,
        ds: datasets.Dataset,
    ) -> dict[str, float]:

        tagged_ds = self.predict(ds)
        return eval.get_f1_scores(tagged_ds["predicted_labels"], tagged_ds["label_ids"])
