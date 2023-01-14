import haystack as hs
import pandas as pd

SIMULATED_TOP_K_READER = 1


def retriever_recall(
    retriever: hs.nodes.BaseRetriever,
    labels_agg: list[hs.MultiLabel],
    top_ks: list[int] = [1, 3, 5, 10, 20],
) -> pd.DataFrame:
    pipe = hs.pipelines.DocumentSearchPipeline(retriever)
    eval_results = pipe.eval(
        labels=labels_agg, params={"Retriever": {"top_k": max(top_ks)}}
    )

    results = {}
    for k in top_ks:
        metrics = eval_results.calculate_metrics(simulated_top_k_retriever=k)
        results[k] = {"recall": metrics["Retriever"]["recall_single_hit"]}

    return pd.DataFrame.from_dict(results, orient="index")


def evaluate_reader(
    reader: hs.nodes.BaseReader,
    labels_agg: list[hs.MultiLabel],
    score_keys: list[str] = None,
) -> dict[str, float]:

    if score_keys is None:
        score_keys = ["exact_match", "f1"]

    pipe = hs.pipelines.Pipeline()
    pipe.add_node(component=reader, name="Reader", inputs=["Query"])
    docs = [
        [label.document for label in multilabel.labels] for multilabel in labels_agg
    ]
    eval_result = pipe.eval(labels=labels_agg, documents=docs, params={})
    metrics = eval_result.calculate_metrics(
        simulated_top_k_reader=SIMULATED_TOP_K_READER
    )
    return {k: v for k, v in metrics["Reader"].items() if k in score_keys}
