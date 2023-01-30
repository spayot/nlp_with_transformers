from typing import Any

import datasets
import numpy as np
import torch
import transformers as tfm

from . import eval


class TransformerWithMeanPooling(torch.nn.Module):
    def __init__(self, model: tfm.AutoModel):
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
    def __init__(self, train_ds: datasets.Dataset):
        self.train_ds = train_ds

    def get_neighbor_tags(self, example, k: int):
        neighbors = self.train_ds.get_nearest_examples(
            "embeddings", query=np.array(example["embeddings"]), k=k
        )
        return neighbors.examples["label_ids"]

    @staticmethod
    def select_tags_from_neighbor_tags(
        neighbor_labels: list[list[int]], threshold: int
    ) -> list[int]:
        return list((np.array(neighbor_labels).sum(axis=0) >= threshold) * 1)

    def tag_from_neighbors(self, example: dict[str, list[Any]], k: int, threshold: int):
        assert (
            threshold <= k
        ), f"threshold {threshold} should be lower or equal to k ({k})"

        neighbor_labels = self.get_neighbor_tags(example, k)
        return {
            "predicted_labels": self.select_tags_from_neighbor_tags(
                neighbor_labels, threshold
            )
        }
    
    def score(
        self, 
        valid_ds: datasets.Dataset, 
        k: int, 
        threshold: int,
    ) -> dict[str, float]:
        
        tagged_ds = valid_ds.map(
            self.tag_from_neighbors, 
            fn_kwargs={"k": k, "threshold": threshold},
            )

        return eval.get_f1_scores(tagged_ds["predicted_labels"], tagged_ds["label_ids"])
    