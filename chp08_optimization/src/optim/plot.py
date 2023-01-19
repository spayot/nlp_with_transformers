import matplotlib.pyplot as plt
import pandas as pd

from .perf import PerfMetrics


def plot_metrics(
    perf_metrics: PerfMetrics,
    current_optim_type: str,
) -> None:
    """scatter plot model performance with accuracy as y axis, latency as x axis and
    model size as point size."""
    df = pd.DataFrame.from_dict(perf_metrics.metrics, orient="index")

    fig, _ = plt.subplots()
    scatter_plot_df(df, current_optim_type)

    _ = plt.legend(bbox_to_anchor=(1, 1))
    # for handle in legend.legendHanles:
    #     handle.set_sizes([20])

    plt.ylim(80, 90)  # (*ylim)
    plt.xlim(1, df.latency_ms_avg.max() + 3)
    plt.title("Performance Benchmark (bubble size = model size)")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Average Latency (ms)")
    plt.show()

    return fig


def scatter_plot_df(df: pd.DataFrame, current_optim_type: str):
    for idx in df.index:
        df_opt = df.loc[idx]
        # Add a dashed circle around the current opitmization type
        kwargs = {"marker": "$\u25CC$"} if idx == current_optim_type else {}
        plt.scatter(
            df_opt["latency_ms_avg"],
            df_opt["accuracy"] * 100,
            alpha=0.5,
            s=df_opt["size"],
            label=idx,
            **kwargs,
        )
