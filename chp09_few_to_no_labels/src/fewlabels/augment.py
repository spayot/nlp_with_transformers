from typing import Any, Protocol

# refactored
from .preproc import Batch


class AugmentBatchFcn(Protocol):
    def __call__(self, batch: Batch) -> Batch:
        """callback protocol that returns batch examples + a certain number
        of augmentations for each of those examples.
        output batch will typically have size: (input batch size) * (1 + n)
            where n is an integer number of transformations"""
        ...


class AugmentFcn(Protocol):
    def __call__(self, text: str) -> str:
        ...


def augment_batch(
    batch: Batch, augment_fcn: AugmentFcn, n_transformations: int = 1
) -> Batch:
    """implements"""
    augmented_labels, augmented_texts = [], []
    for text, label_ids in zip(batch["text"], batch["label_ids"]):
        t, l = augment_single_sample(augment_fcn, text, label_ids, n_transformations)
        augmented_labels += l
        augmented_texts += t
    return {"text": augmented_texts, "label_ids": augmented_labels}


def augment_single_sample(
    augment_fcn: AugmentFcn,
    text: str,
    label_ids: list[int],
    n_transformations: int,
) -> tuple[list[Any]]:
    augmented_labels = [label_ids for _ in range(n_transformations + 1)]
    augmented_texts = [text] + [
        augment_fcn(text) for _ in range(n_transformations)
    ]
    return augmented_texts, augmented_labels
