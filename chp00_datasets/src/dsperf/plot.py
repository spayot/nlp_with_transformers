import matplotlib.pyplot as plt
import pandas as pd


def filter_dataframe_when_index_contains(
    df: pd.DataFrame, index_contains: str
) -> pd.DataFrame:
    return df.loc[[idx for idx in df.index if index_contains in idx]]


def plot_test_type(df: pd.DataFrame, test_type: str, ax):
    df_to_plot = filter_dataframe_when_index_contains(df, test_type).T
    df_to_plot.plot.barh(ax=ax, title=test_type, stacked=True)
    plt.xticks(rotation=0)


def plot_test_results(test_results: dict[str, dict]):
    df = pd.DataFrame(test_results)

    fig, ax = plt.subplots(2, 1)
    plot_test_type(df, "memory", ax[0])
    plot_test_type(df, "cpu", ax[1])
    # ax[1].legend([])
    plt.show()

    return fig
