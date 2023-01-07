import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds: np.ndarray, y_true: np.ndarray, labels: np.ndarray):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()


def plot_single_hexmap(df: pd.DataFrame, label: str, cmap: str, ax):
    """plots a single hexmap"""
    ax.hexbin(df["X"], df["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    ax.set_title(label, c="black")
    ax.set_xticks([])
    ax.set_yticks([])


def gridplot_hexmaps(
    labels: list[str], cmaps: list[str], df: pd.DataFrame, shape: tuple = (2, 3)
):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_sub = df.query(f"label == {i}")
        plot_single_hexmap(df_sub, label, cmap, axes[i])

    plt.tight_layout()
    plt.show()
