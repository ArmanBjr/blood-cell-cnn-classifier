"""Plotting utilities for EDA and training history."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
COLORS = ["#4FA3D1", "#5DBB8A", "#E8A838", "#D15F4F"]


def plot_sample_grid(generator, n_per_class: int = 8, save_path: str = None):
    """Show n_per_class sample images for each class in a grid."""
    class_to_idx = generator.class_indices
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    samples = {cls: [] for cls in CLASS_NAMES}
    for img_path, label in zip(generator.filepaths, generator.classes):
        cls = idx_to_class[label]
        if len(samples[cls]) < n_per_class:
            samples[cls].append(img_path)
        if all(len(v) == n_per_class for v in samples.values()):
            break

    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class, figsize=(n_per_class * 2, len(CLASS_NAMES) * 2))
    for r, cls in enumerate(CLASS_NAMES):
        for c, img_path in enumerate(samples[cls]):
            img = plt.imread(img_path)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(cls, fontsize=9, color=COLORS[r], loc="left")
    plt.suptitle("Sample Images per Class", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_class_distribution(generator, save_path: str = None):
    """Bar chart of class counts."""
    class_to_idx = generator.class_indices
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = generator.classes
    counts = [np.sum(labels == i) for i in range(len(CLASS_NAMES))]
    names = [idx_to_class[i] for i in range(len(CLASS_NAMES))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, counts, color=COLORS)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(count), ha="center", va="bottom")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Images")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(history, model_name: str, save_path: str = None):
    """Plot accuracy and loss curves from a Keras History object."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(history.history["accuracy"], label="Train", color=COLORS[0])
    ax1.plot(history.history["val_accuracy"], label="Val", color=COLORS[1])
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train", color=COLORS[0])
    ax2.plot(history.history["val_loss"], label="Val", color=COLORS[1])
    ax2.set_title(f"{model_name} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None):
    """Bar chart comparing test accuracy of all models."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=COLORS * 3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{acc:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Comparison — Test Accuracy")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
