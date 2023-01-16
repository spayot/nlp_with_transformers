import datasets
import pandas as pd
import transformers as tfm


def tag_text(
    text: str,
    model: tfm.AutoModelForTokenClassification,
    tokenizer: tfm.AutoTokenizer,
    tags: datasets.ClassLabel,
) -> pd.DataFrame:

    tokens = tokenizer(text).tokens()

    input_ids = tokenizer.encode(text, return_tensors="pt")

    outputs = model(input_ids)

    predictions = outputs.logits[0].argmax(-1)

    preds = [tags.names[p] for p in predictions]

    return pd.DataFrame({"tokens": tokens, "tags": preds}).T
