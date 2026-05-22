import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import GLOBAL_SEED, SPLIT_SIZE, SPLITS
from transforms import get_vgg_transforms, get_simclr_transforms


class NucleiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = ["Tumor", "Lymphocyte", "Histiocyte"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # iterate over classes and load samples for each class
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue

            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        # count samples per class
        for cls in self.classes:
            count = sum(
                1 for _, label in self.samples if label == self.class_to_idx[cls]
            )
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# for reproducability, Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_vgg_loaders(data_dir, batch_size=32, num_workers=4, seed=GLOBAL_SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    data_dir = Path(data_dir)

    train_set = NucleiDataset(
        data_dir / "train", transform=get_vgg_transforms(training=True)
    )
    val_set = NucleiDataset(
        data_dir / "val", transform=get_vgg_transforms(training=False)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


MAX_SAMPLES_CONTRASTIVE_PER_CLASS = 2500


class NucleiContrastiveLearningDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="pretrain"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.images = []
        self.labels = []
        self.split = split
        self.classes = ["Tumor", "Lymphocyte", "Histiocyte"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        assert split in SPLITS

        # iterate over classes and load samples for each class
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue

            filepaths = list(class_dir.glob("*.png"))

            # for finetuning and validation, limit the number of max samples per class
            if (
                split != "pretrain"
                and len(filepaths) > MAX_SAMPLES_CONTRASTIVE_PER_CLASS
            ):
                filepaths = random.sample(filepaths, MAX_SAMPLES_CONTRASTIVE_PER_CLASS)

            for img_path in filepaths:
                self.samples.append((img_path, self.class_to_idx[class_name]))
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

        # count samples per class
        for cls in self.classes:
            count = sum(
                1 for _, label in self.samples if label == self.class_to_idx[cls]
            )
            print(f"  {cls}: {count}")

        if split != "pretrain":
            # split validation data into train and val sets (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                self.images,
                self.labels,
                test_size=(SPLIT_SIZE),
                random_state=GLOBAL_SEED,
            )

            if split == "finetune":
                self.images = X_train
                self.labels = y_train

            else:
                self.images = X_test
                self.labels = y_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform and self.split == "pretrain":
            # generate two augmentated images for pretraining
            image = [self.transform(image) for _ in range(2)]

        elif self.transform:
            image = self.transform(image)

        return image, label


def create_simclr_loader(
    data_dir,
    batch_size=32,
    num_workers=4,
    seed=GLOBAL_SEED,
    split="pretrain",
    drop_last=False,
):
    g = torch.Generator()
    g.manual_seed(seed)

    dataset = NucleiContrastiveLearningDataset(
        data_dir, transform=get_simclr_transforms(split=split), split=split
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "pretrain",
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=drop_last,
    )

    return dataloader


if __name__ == "__main__":
    data_dir = "nuclei_patches"

    print("Loading data...")
    train_loader, val_loader = create_vgg_loaders(data_dir, batch_size=32)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels[:5]}")
