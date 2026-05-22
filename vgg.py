import time
import torch
import torch.nn as nn
from pathlib import Path

from utils import (
    train_model,
    save_model_weights,
    load_model_weights,
    to_json,
    init_seed,
    count_model_trainable_params,
)
from plots import plot_curves

from evaluation import evaluate
from dataloader import create_vgg_loaders

from config import VGG_BATCH_SIZE, VGG_NUM_EPOCHS


class NucleiNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        conv_layer_sizes = [3, 32, 64, 128, 256, 512]
        fc_layer_sizes = [512, 2048]

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()

        for i in range(len(conv_layer_sizes)):
            in_channels = conv_layer_sizes[i]

            # bound checking
            if (i + 1) < len(conv_layer_sizes):
                out_channels = conv_layer_sizes[i + 1]
            else:
                out_channels = conv_layer_sizes[-1]

            # add conv layer
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                )
            )

            # add relu and pool layers
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(len(fc_layer_sizes)):
            is_last_layer = False

            in_features = fc_layer_sizes[i]

            # bound checking
            if (i + 1) < len(fc_layer_sizes):
                out_features = fc_layer_sizes[i + 1]
            else:
                out_features = num_classes
                is_last_layer = True

            self.fc_layers.append(
                nn.Linear(in_features=in_features, out_features=out_features)
            )

            # add relu and dropout if not last layer
            if not is_last_layer:
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout())

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = torch.flatten(x, 1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        return x


def create_model():
    model = NucleiNet(num_classes=3)

    return model


def load_model(filepath):
    model = create_model()

    model = load_model_weights(model, filepath)

    return model


if __name__ == "__main__":
    init_seed()

    train_loader, val_loader = create_vgg_loaders(
        "./nuclei_patches/", batch_size=VGG_BATCH_SIZE
    )

    model = create_model()

    num_params = count_model_trainable_params(model)

    print(f"\nNo of trainable params: {num_params / 1e6:.2f}M")

    print(f"Training...\n")
    # train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader=train_loader, val_loader=val_loader
    )

    print(f"\nEvaluating...")

    # evaluate model
    test_acc, overall_classification_report, overall_confusion_matrix = evaluate(model)
    _, primary_classification_report, primary_confusion_matrix = evaluate(
        model, sample_type="primary"
    )
    _, metastatic_classification_report, metastatic_confusion_matrix = evaluate(
        model, sample_type="metastatic"
    )

    epoch_time = int(time.time())
    output_dir = Path(f"res/nuclei-net/{epoch_time}-test-acc-{test_acc:.2f}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTrain acc = {train_accs[-1]:.2f}%, Val acc = {val_accs[-1]:.2f}%")
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
    to_json(data=train_accs, filepath=f"{output_dir}/train-acc.json")
    to_json(data=val_accs, filepath=f"{output_dir}/val-acc.json")

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
        early_stopping=num_epochs < VGG_NUM_EPOCHS,
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
        early_stopping=num_epochs < VGG_NUM_EPOCHS,
    )

    # save loss and acc plots
    loss_plot.savefig(output_dir / "vgg-loss-plot.png")
    acc_plot.savefig(output_dir / "vgg-acc-plot.png")

    # save model weights
    save_model_weights(model, filepath=f"{output_dir}/weights.pth")

    print(f"Results saved to {output_dir}\n")
