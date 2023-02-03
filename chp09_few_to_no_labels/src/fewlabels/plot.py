import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval import SlicedTrainingEvaluator


def plot_metrics(evaluator: SlicedTrainingEvaluator, current_model: str) -> None:
    """plot f1 scores for each model, depending on # of training samples"""
    sample_sizes = [len(slice) for slice in evaluator.train_slices]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for run in evaluator.f1_scores["micro"]:
        if run == current_model:
            for ax, score_type in zip(axs, evaluator.f1_scores):
                ax.plot(
                    sample_sizes,
                    evaluator.f1_scores[score_type][run],
                    label=run,
                    linewidth=2,
                )
        else:
            for ax, score_type in zip(axs, evaluator.f1_scores):
                ax.plot(
                    sample_sizes,
                    evaluator.f1_scores[score_type][run],
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

    return fig


def plot_gridsearch(micro_scores: np.ndarray, macro_scores: np.ndarray):
    """used for hyperparameters gridsearch for embedding-based lookup tables.
    generates 2-D visualizations for micro and macro scores based on choice for
    - k: number of neighbors
    - threshold: minimum number of neighbors having a label for that label to be attributed to
        query example"""
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax.flatten()
    plot_single_gridsearch(micro_scores, ax[0], "micro scores")
    plot_single_gridsearch(macro_scores, ax[1], "macro scores")

    plt.tight_layout()

    return fig


def plot_single_gridsearch(scores: np.ndarray, ax, title: str):
    ax.imshow(scores)
    ax.set_xlabel("threshold")
    ax.set_ylabel("k")
    ax.set_title(title)
    ax.set_xlim([0.5, len(scores) - 1.5])
    ax.set_ylim([len(scores) - 1.5, 0.5])


def plot_trainer_loss(log_history):
    df_log = pd.DataFrame(log_history)
    field_map = {
        "eval_loss": "Validation",
        "loss": "Train",
    }
    for loss_type, label in field_map.items():
        (df_log.dropna(subset=loss_type).reset_index()[loss_type].plot(label=label))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
