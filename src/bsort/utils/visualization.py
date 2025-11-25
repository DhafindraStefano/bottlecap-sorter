import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from PIL import Image


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Plots a confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        title: Plot title.

    Returns:
        PIL.Image: The plot as an image object.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def plot_class_distribution(class_counts, class_names, title="Class Distribution"):
    """
    Plots class distribution bar chart.

    Args:
        class_counts: Dictionary of class ID to count.
        class_names: Dictionary or list mapping class ID to name.
        title: Plot title.

    Returns:
        PIL.Image: The plot as an image object.
    """
    data = []
    for cls_id, count in class_counts.items():
        name = (
            class_names[cls_id]
            if isinstance(class_names, dict)
            else class_names[cls_id]
        )
        data.append({"Class": name, "Count": count})

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Class", y="Count", data=df, palette="viridis")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def log_plots_to_wandb(plots_dict):
    """
    Logs a dictionary of PIL images to wandb.

    Args:
        plots_dict (dict): Dictionary mapping plot names to PIL images.
    """
    if wandb.run is None:
        return

    wandb_images = {k: wandb.Image(v) for k, v in plots_dict.items()}
    wandb.log(wandb_images)
