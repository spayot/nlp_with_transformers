import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds: np.ndarray, y_true: np.ndarray, labels: np.ndarray):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    _, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()


def plot_f1_scores(f1_scores: dict[str, dict], save_path: str = None):
    fig, ax = plt.subplots()
    pd.DataFrame(f1_scores).plot.barh(title="NER f1-score", ax=ax)
    plt.legend(["only de training data", "all-languages"])
    for k, scores in f1_scores.items():
        delta = -0.12 if k == "de" else +0.12
        for y, score in zip(ax.get_yticks(), scores.values()):
            ax.text(score, y + delta, f"{score:.3}")

    plt.ylabel("language")

    if save_path:
        plt.savefig(save_path)

    plt.show()
