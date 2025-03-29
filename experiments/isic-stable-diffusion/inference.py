import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ["PROJECT_ROOT"]
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from diffusion.stable_diffusion_classifier import StableDiffusionClassifier
from dataset.isic import ISICDataLoader
from classifier.classifier import Classifier
from utils.metrics import Accuracy, F1, Precision, Recall

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate


class InferenceConfig:
    def __init__(self):
        config_str = os.environ.get("CONFIG")
        if config_str is None:
            raise ValueError("CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config["project_root"]
        self.experiment_dir = self.config["experiment_dir"]
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)

def label_to_text_mapper(labels:torch.tensor):
    assert len(labels.shape) == 1, "Labels should be a 1D tensor"
    output = []
    
    for l in range(labels.shape[0]):
        label = labels[l].item()
        if label == 0:
            output.append("image of a benign skin lesion")
        elif label == 1:
            output.append("image of a melanoma skin lesion")
        else:
            raise ValueError("Invalid label")
    return output

def main(steps=51):
    global config
    config = InferenceConfig()
    config.evaluation_per_stage = [steps]

    # Set random seed
    accelerate.utils.set_seed(config.seed)

    data = ISICDataLoader(
        wavelet_transform=config.wavelet_transform,
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    test_loader = data.get_test_loader()

    classifier = StableDiffusionClassifier(config, label_to_text_mapper)

    metrics = [
        Accuracy("accuracy"),
        F1("f1"),
        Precision("precision"),
        Recall("recall")
    ]

    metric_output, _, _ = classifier.inference(val_loader, metrics)
    print([{k: round(v.item(), 4) for k, v in d.items()} for d in metric_output])

if __name__ == "__main__":
    main()
