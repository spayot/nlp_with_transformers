import haystack as hs
import pandas as pd


def create_labels_from_df(df: pd.DataFrame) -> list[hs.Label]:
    labels = []
    for _, row in df.iterrows():
        labels += create_labels_from_row(row)

    return labels


def create_labels_from_row(row: pd.Series) -> list[hs.Label]:
    if len(row["answers.text"]) == 0:
        return [create_label(row, answer="", no_answer=True)]
    return [
        create_label(row, answer, no_answer=False) for answer in row["answers.text"]
    ]


def create_label(row: pd.Series, answer: str, no_answer: bool):
    meta = {"item_id": row["title"], "question_id": row["id"]}
    return hs.Label(
        query=row["question"],
        answer=hs.Answer(answer=answer),
        origin="gold-label",
        document=hs.Document(content=row["context"], id=row["review_id"]),
        meta=meta,
        is_correct_answer=True,
        is_correct_document=True,
        no_answer=no_answer,
        filters={"item_id": [row["title"]], "split": ["test"]},
    )
