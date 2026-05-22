import torchvision.transforms as transforms
from config import IMG_SIZE, ROTATION_FACTOR, normalisation_transform


def get_vgg_transforms(training=True):
    if training:
        # Data augmentation for training
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(IMG_SIZE),
                transforms.RandomRotation(ROTATION_FACTOR),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                normalisation_transform,
            ]
        )
    else:
        # Just normalization for validation/test
        return transforms.Compose(
            [transforms.Resize(96), transforms.ToTensor(), normalisation_transform]
        )


def get_simclr_transforms(split="pretrain"):
    if split == "pretrain":
        # Data augmentation for pretraining
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomRotation(ROTATION_FACTOR),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomGrayscale(),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                normalisation_transform,
            ]
        )
    else:
        # Just normalization for finetuning/validation
        return transforms.Compose(
            [transforms.Resize(96), transforms.ToTensor(), normalisation_transform]
        )
