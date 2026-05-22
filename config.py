import torchvision.transforms as transforms

GLOBAL_SEED = 0

VGG_BATCH_SIZE = 256
VGG_NUM_EPOCHS = 75

TEMPERATURE = 0.1
IMG_SIZE = 96
PRETRAINING_BATCH_SIZE = 512
PRETRAINING_EPOCHS = 100
FINETUNING_EPOCHS = 50
FINETUNING_BATCH_SIZE = 256

SPLIT_SIZE = 0.8
SPLITS = ["pretrain", "finetune", "val"]

ROTATION_FACTOR = 90

# normalization values
normalisation_transform = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],  # R, G, B means
    std=[0.5, 0.5, 0.5],  # R, G, B sd
)


# Path to the data directory
DATA_DIR = "./nuclei_patches/"

# Path to the test set directory
TEST_SET_DIR = "./test_set/"

