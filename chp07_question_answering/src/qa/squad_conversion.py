import json

import pandas as pd


def convert_all_data_to_squad(
    dfs: dict[str, pd.DataFrame], fpath_template: str
) -> None:
    for split, df in dfs.items():
        save_df_to_squad_json(df, fpath_template.format(split=split))


def save_df_to_squad_json(df: pd.DataFrame, fpath: str) -> None:
    with open(fpath, "w+", encoding="utf-8") as f:
        json.dump(convert_df_to_squad(df), f)


def convert_df_to_squad(df: pd.DataFrame) -> dict:
    return {
        "data": [
            {
                "title": title,
                "paragraphs": create_all_paragraphs_per_title(
                    df.query(f"title=='{title}'")
                ),
            }
            for title in df.title.unique()
        ]
    }


def create_all_paragraphs_per_title(df_by_title: pd.DataFrame):
    review_ids = df_by_title.review_id.unique()
    return [
        create_paragraph(df_by_title.query(f"review_id=='{review_id}'"))
        for review_id in review_ids
    ]


def create_paragraph(df: pd.DataFrame) -> dict:
    paragraph = {
        "context": df.context.iloc[0],  # all rows should have same context
        "qas": [create_qas(row) for _, row in df.iterrows()],
    }
    return paragraph


def create_qas(row):
    return {
        "question": row["question"],
        "id": row["id"],
        "answers": create_answers_from_row(row),
    }


def create_answers_from_row(row) -> dict:
    if len(row["answers.text"]) == 0:
        return []
    return [
        {"text": text, "answer_start": int(start)}
        for text, start in zip(row["answers.text"], row["answers.answer_start"])
    ]
