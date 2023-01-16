import transformers as tfm

# pytorch knows to exclude tokens from loss calculation when classification label is -100
IGNORED_TAG_ID = -100


def align_tags(word_ids: list[int], ner_tags: list[int]) -> list[int]:
    """creates NER tags at the token level, from a list of NER tags at the word level.
    only first token of a split word is assigned a label, subsequent ones are assigned -100.
    special tokens are also assigned -100.
    """
    aligned_tags = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id == previous_word_id or word_id == None:
            aligned_tags.append(IGNORED_TAG_ID)
        else:
            aligned_tags.append(ner_tags[word_id])
        previous_word_id = word_id

    return aligned_tags


def tokenize_and_align_labels(
    examples: dict[str, list],
    tokenizer: tfm.AutoTokenizer,
) -> dict[str, list]:

    # tokenize examples
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    # align tags to tokens, based on word ids
    aligned_tags = []
    for idx, labels in enumerate(examples["ner_tags"]):
        aligned_tags.append(
            align_tags(
                word_ids=tokenized_inputs.word_ids(batch_index=idx),
                ner_tags=labels,
            )
        )

    tokenized_inputs["labels"] = aligned_tags

    return tokenized_inputs
