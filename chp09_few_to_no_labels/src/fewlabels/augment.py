from typing import Any

import nlpaug.augmenter.word as naw

# refactored
Batch = dict[str, list[Any]]


def augment_batch(
    batch: Batch,
    augmenter: naw.ContextualWordEmbsAug,
    n_transformations: int = 1,
) -> dict[str, list[Any]]:
    augmented_labels, augmented_texts = [], []
    for text, label_ids in zip(batch["text"], batch["label_ids"]):
        t, l = augment_single_sample(augmenter, text, label_ids, n_transformations)
        augmented_labels += l
        augmented_texts += t
    return {"text": augmented_texts, "label_ids": augmented_labels}


def augment_single_sample(
    augmenter: naw.ContextualWordEmbsAug,
    text: str,
    label_ids: list[int],
    n_transformations: int,
) -> tuple[list[Any]]:
    augmented_labels = [label_ids for _ in range(n_transformations + 1)]
    augmented_texts = [text] + [
        augmenter.augment(text) for _ in range(n_transformations)
    ]
    return augmented_texts, augmented_labels
