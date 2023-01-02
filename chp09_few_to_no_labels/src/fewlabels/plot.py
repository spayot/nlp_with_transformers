import matplotlib.pyplot as plt


def plot_metrics(
    f1_scores: dict[str, dict], sample_sizes: list[int], current_model: str
) -> None:
    """plot f1 scores for each model, depending on # of training samples"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for run in f1_scores["micro"]:
        if run == current_model:
            for ax, score_type in zip(axs, f1_scores):
                ax.plot(
                    sample_sizes, f1_scores[score_type][run], label=run, linewidth=2
                )
        else:
            for ax, score_type in zip(axs, f1_scores):
                ax.plot(
                    sample_sizes,
                    f1_scores[score_type][run],
                    label=run,
                    linestyle="dashed",
                )

    axs[0].set_title("Micro F1 Scores")
    axs[1].set_title("Macro F1 Scores")
    axs[0].set_ylabel("Test set F1 Score")
    axs[0].legend(loc="lower right")

    for ax in axs:
        ax.set_xlabel("number of training samples")
        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)
    plt.tight_layout()
    plt.show()
