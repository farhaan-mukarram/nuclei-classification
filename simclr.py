import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    count_model_trainable_params,
    init_seed,
    load_model_weights,
    save_model_weights,
    to_json,
    get_device,
    train_model,
)
from vgg import NucleiNet
from evaluation import evaluate

from dataloader import create_simclr_loader
from plots import plot_curves, plot_tsne
from config import (
    TEMPERATURE,
    FINETUNING_EPOCHS,
    PRETRAINING_EPOCHS,
    FINETUNING_BATCH_SIZE,
    PRETRAINING_BATCH_SIZE,
)


def create_simclr_model():
    model = NucleiNet(num_classes=128)
    model.fc_layers = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=128)
    )

    return model


# Source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
def info_nce_loss(feats):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # InfoNCE loss
    cos_sim = cos_sim / TEMPERATURE
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll


# function to pretrain the encoder
def pretrain(model, train_loader):
    device = get_device()
    model = model.to(device)

    train_losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005)

    num_params = count_model_trainable_params(model)
    print(f"\nNo of trainable params (pre-training): {num_params / 1e6:.2f}M")
    print("Pretraining...\n")

    for epoch in range(PRETRAINING_EPOCHS):
        train_loss = 0

        for batch in train_loader:
            images, _ = batch
            images = torch.cat(images, dim=0)

            images = images.to(device)

            optimizer.zero_grad()
            features = model(images)
            loss = info_nce_loss(features)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{PRETRAINING_EPOCHS}: Train Loss {train_loss}")

    print("\nFinished pretraining")

    return model, train_losses


# function to finetune the classification head
def finetune(model, train_loader, val_loader):
    num_epochs = FINETUNING_EPOCHS

    num_params = count_model_trainable_params(model)
    print(f"No of trainable params (fine-training): {num_params}")
    print("Finetuning...\n")

    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs
    )

    return model, train_losses, val_losses, train_accs, val_accs


def create_finetuned_model(model):
    # freeze prior layers
    for param in model.parameters():
        param.requires_grad = False

    # add classification head
    model.fc_layers.append(torch.nn.Linear(in_features=128, out_features=3))

    return model


def load_model(filepath):
    model = create_simclr_model()

    model = load_model_weights(model, filepath=filepath)

    return model


if __name__ == "__main__":
    data_dir = "nuclei_patches_contrastive"

    data_dir = Path(data_dir)

    init_seed()

    pretrain_loader = create_simclr_loader(
        data_dir / "train", batch_size=PRETRAINING_BATCH_SIZE, drop_last=True
    )
    finetune_loader = create_simclr_loader(
        data_dir / "val", batch_size=FINETUNING_BATCH_SIZE, split="finetune"
    )
    val_loader = create_simclr_loader(
        data_dir / "val", batch_size=FINETUNING_BATCH_SIZE, split="val"
    )

    model = create_simclr_model()

    # pretrain model
    model, train_losses = pretrain(model, pretrain_loader)

    # save pretraining results
    epoch_time = int(time.time())
    output_dir = Path(f"res/nuclei-net-simclr/{epoch_time}/pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    to_json(data=train_losses, filepath=f"{output_dir}/train-loss.json")
    save_model_weights(model, filepath=f"{output_dir}/weights.pth")

    tsne_plot = plot_tsne(model, finetune_loader)

    tsne_plot.savefig(output_dir / "simclr-tsne-plot.png", bbox_inches="tight")

    print(f"Pretraining results saved to {output_dir}\n")

    # fine-tune model
    finetuned_model = create_finetuned_model(model)

    finetuned_model, train_losses, val_losses, train_accs, val_accs = finetune(
        model, finetune_loader, val_loader
    )

    print(f"\nEvaluating...")

    # evaluate model
    test_acc, overall_classification_report, overall_confusion_matrix = evaluate(
        finetuned_model
    )
    _, primary_classification_report, primary_confusion_matrix = evaluate(
        finetuned_model, sample_type="primary"
    )
    _, metastatic_classification_report, metastatic_confusion_matrix = evaluate(
        finetuned_model, sample_type="metastatic"
    )

    output_dir = Path(
        f"res/nuclei-net-simclr/{epoch_time}/finetune/test-acc-{test_acc:.2f}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train acc = {train_accs[-1]:.2f}%, Val acc = {val_accs[-1]:.2f}%")
    print(f"Test acc = {test_acc:.2f}%\n")

    # save classification report to file
    with open(output_dir / "classification_report.txt", "w") as f:
        print("=" * 70, file=f)
        print("OVERALL PERFORMANCE", file=f)
        print("=" * 70, file=f)
        print(overall_classification_report, file=f)

        print("=" * 70, file=f)
        print("PRIMARY SAMPLE PERFORMANCE", file=f)
        print("=" * 70, file=f)
        print(primary_classification_report, file=f)

        print("=" * 70, file=f)
        print("METASTATIC SAMPLE PERFORMANCE", file=f)
        print("=" * 70, file=f)
        print(metastatic_classification_report, file=f)

    # save confusion matrices
    overall_confusion_matrix.figure_.savefig(
        output_dir / "overall_confusion_matrix.png", bbox_inches="tight"
    )

    primary_confusion_matrix.figure_.savefig(
        output_dir / "primary_confusion_matrix.png", bbox_inches="tight"
    )

    metastatic_confusion_matrix.figure_.savefig(
        output_dir / "metastatic_confusion_matrix.png", bbox_inches="tight"
    )

    # save losses and accs
    to_json(data=train_losses, filepath=f"{output_dir}/train-loss.json")
    to_json(data=val_losses, filepath=f"{output_dir}/val-loss.json")
    to_json(data=val_accs, filepath=f"{output_dir}/val-acc.json")
    to_json(data=train_accs, filepath=f"{output_dir}/train-acc.json")

    # plot loss and accuracy curves and save
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    # plot loss curve
    loss_plot = plot_curves(
        x=epochs,
        y1=train_losses,
        y2=val_losses,
        y1_label="Train Loss",
        y2_label="Val Loss",
        x_label="Epochs",
        y_label="Loss",
        early_stopping=num_epochs < FINETUNING_EPOCHS,
    )

    # plot acc curve
    acc_plot = plot_curves(
        x=epochs,
        y1=train_accs,
        y2=val_accs,
        y1_label="Train Acc.",
        y2_label="Val Acc.",
        x_label="Epochs",
        y_label="Accuracy",
        early_stopping=num_epochs < FINETUNING_EPOCHS,
    )

    # save loss and acc plots
    loss_plot.savefig(output_dir / "simclr-loss-plot.png")
    acc_plot.savefig(output_dir / "simclr-acc-plot.png")

    # save model weights
    save_model_weights(finetuned_model, filepath=f"{output_dir}/weights.pth")

    print(f"Finetuning results saved to {output_dir}\n")
