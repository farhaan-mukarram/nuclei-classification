from sklearn.manifold import TSNE
from config import GLOBAL_SEED
import matplotlib.pyplot as plt

import numpy as np
from utils import get_device
import torch


# function to plot the t-SNE plot for a model
def plot_tsne(model, data_loader):
    latents = []
    y_true = []

    device = get_device()

    model.eval()
    model = model.to(device)

    # generate latent representation
    with torch.no_grad():
        for val_data in data_loader:
            inputs, labels = val_data
            inputs = inputs.to(device)
            labels = labels.to(device).cpu().numpy()

            y_true.append(labels)

            # forward
            outputs = model(inputs).cpu().numpy()
            latents.append(outputs)

    latents = np.concat(latents, axis=0)
    y_true = np.concat(y_true, axis=0)
    fig, ax = plt.subplots()

    # Plot latents using t-SNE
    # Source: https://medium.com/@girishajmera/autoencoders-tsne-exploratory-data-analysis-on-unlabeled-image-dataset-3bdf499dbad3
    tsne = TSNE(n_components=2, random_state=GLOBAL_SEED, metric="cosine")
    tsne_features = tsne.fit_transform(latents)

    ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_true, alpha=0.5)

    ax.set_axis_off()

    return fig


# function to plot loss/acc curves for a model
def plot_curves(x, y1, y2, y1_label, y2_label, x_label, y_label, early_stopping=True):
    fig, ax = plt.subplots()

    ax.plot(x, y1, label=y1_label)
    ax.plot(x, y2, label=y2_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True, alpha=0.3)

    # add dashed line to indicate early stopping, last 10 epochs
    if early_stopping:
        ax.axvline(x=x[-10], color="red", linestyle="--")

    ax.legend()
    fig.tight_layout()

    return fig
