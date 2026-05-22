import numpy as np
import re
import pathlib
import torch
from torchvision import transforms
from PIL import Image
import sklearn.metrics

from utils import get_device
from config import IMG_SIZE, normalisation_transform, TEST_SET_DIR

transform = transforms.Compose(
    [transforms.Resize(IMG_SIZE), transforms.ToTensor(), normalisation_transform]
)

classes = ["tumor", "lymphocyte", "histiocyte"]
sample_types = ["primary", "metastatic"]
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# nuclei_ followed by tumor, lymphocyte or histiocyte
CLASS_REGEX = re.compile(r"nuclei_([a-zA-z]+)_")


# function to evaluate a model on the provided test set.
def evaluate(model, sample_type=None):
    if sample_type is not None:
        assert sample_type in sample_types

    p = pathlib.Path(TEST_SET_DIR)

    device = get_device()

    model = model.to(device)
    model.eval()

    #
    paths = sorted([path for path in p.glob("*.npy")])

    labels = []
    preds = []

    correct = 0

    with torch.no_grad():
        for path in paths:
            array = np.load(path)
            image = Image.fromarray(array)
            image = transform(image)
            image = image.to(device)

            # extract class from path
            stem = path.stem

            # sample type is either primary or metastatic, filter accordingly
            if sample_type is not None and sample_type not in stem:
                continue

            # get matched classname
            match = re.search(CLASS_REGEX, stem)

            if match is None:
                continue

            actual_class = match.group(1)
            labels.append(classes.index(actual_class))

            # forward pass through the model
            image = image.unsqueeze(0)
            output = model(image)
            output = output.cpu().numpy()[-1]

            # generate prediction
            prediction = np.argmax(output)
            preds.append(prediction)
            predicted_class = classes[prediction]

            if predicted_class == actual_class:
                correct += 1

        # compute overall accuracy
        total = len(labels)
        accuracy = (100 * correct) / total

        # generate classification report
        classification_report = sklearn.metrics.classification_report(
            y_true=labels, y_pred=preds, target_names=classes
        )

        # generate confusion matrix
        cm_display = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=preds,
            normalize="true",
            display_labels=classes,
            colorbar=False,
        )

    return (accuracy, classification_report, cm_display)
