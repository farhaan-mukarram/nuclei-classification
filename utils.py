import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from config import GLOBAL_SEED, VGG_NUM_EPOCHS


# Source - https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        # reset counter if validation loss is less than min
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        # increase counter if validation loss is greater than min + delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1

            # trigger early stopping if count is greater than patience
            if self.counter >= self.patience:
                return True
        return False


# function to train a classification model
def train_model(
    model, train_loader, val_loader, optimizer=None, num_epochs=VGG_NUM_EPOCHS
):
    device = get_device()

    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    model = model.to(device)

    val_losses = []
    train_losses = []

    train_accs = []
    val_accs = []

    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005)

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0

        # compute training loss
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # compute validation loss
        with torch.no_grad():
            for val_data in val_loader:
                inputs, labels = val_data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # compute avg train and validation losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if early_stopper.early_stop(val_loss):
            print("Early Stopping")
            break

        # compute training accuracy
        train_acc = compute_model_accuracy(
            model=model, dataloader=train_loader, device=device
        )

        # compute validation accuracy
        val_acc = compute_model_accuracy(
            model=model, dataloader=val_loader, device=device
        )

        val_losses.append(val_loss)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss {train_loss}, Val Loss : {val_loss}"
        )

    return model, train_losses, val_losses, train_accs, val_accs


# function to save a model's weights to a given filepath
def save_model_weights(model, filepath):
    torch.save(model.state_dict(), filepath)


# function to load model's weights given path to stored weights
def load_model_weights(model, filepath):
    model.load_state_dict(torch.load(filepath, weights_only=True))

    return model


# function to save data as a json file
def to_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)


# function to load data from a saved json file
def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    return data


# Source - https://stackoverflow.com/a/49201237
def count_model_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params


# function to compute a mode's accuracy
def compute_model_accuracy(model, dataloader, device):
    with torch.no_grad():
        correct = 0
        total = 0

        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate outputs
            outputs = model(inputs)
            # get prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = (100 * correct) / total

    return accuracy


# for reproducability (ref: https://docs.pytorch.org/docs/stable/notes/randomness.html)
def init_seed(seed=GLOBAL_SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)


# function to get available device (either CUDA or CPU) for acceleration
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device
