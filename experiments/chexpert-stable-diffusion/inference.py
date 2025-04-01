import sys
import os
import json
import random

# Get project root from environment variable
projectroot = os.environ["PROJECT_ROOT"]
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# Project imports
from diffusion.stable_diffusion_classifier import StableDiffusionClassifier
from dataset.chexpert import CheXpertDataLoader
from classifier.classifier import Classifier
from utils.metrics import Accuracy, F1, Precision, Recall, DPD, EOD

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

        self.sens_attr = self.config["sens_attr"]

    def __getattr__(self, name):
        return self.config.get(name)

def label_to_text_mapper(labels:torch.tensor):
    assert len(labels.shape) == 1, "Labels should be a 1D tensor"
    output = []
    all_ages = list(range(1, 90))
    all_genders = ["male", "female"]
    
    for l in range(labels.shape[0]):
        label = labels[l].item()

        # Select age and gender randomly
        _age = random.choice(all_ages)
        _gender = random.choice(all_genders)

        if label == 0:
            output.append("Chest-Xray of a healthy patient without pleural effusion")
            # output.append("Chest-Xray of a {} healthy patient, aged {}, without pleural effusion".format(_gender, _age))
        elif label == 1:
            output.append("Chest-Xray of a sick patient with pleural effusion")
            # output.append("Chest-Xray of a {} sick patient, aged {}, with pleural effusion".format(_gender, _age))
        else:
            raise ValueError("Invalid label")
    return output

def main():
    global config
    config = InferenceConfig()

    l_to_t_mapper = None
    if(config.sens_attr == 'None'):
        config.sens_attr = None
        l_to_t_mapper = label_to_text_mapper
        metrics = [
            Accuracy("accuracy"),
            F1("f1"),
            Precision("precision"),
            Recall("recall"),
            # DPD("DPD"),
            # EOD("EOD"),
        ]
    else:
        metrics = [
        Accuracy("accuracy"),
        F1("f1"),
        Precision("precision"),
        Recall("recall"),
        DPD("DPD"),
        EOD("EOD"),
    ]
    
    # Set random seed
    accelerate.utils.set_seed(config.seed)

    chexpert = CheXpertDataLoader(
        wavelet_transform=False,
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=(config.image_size, config.image_size),
        augment=False,
        sens_attr=config.sens_attr
    )

    # train_loader = chexpert.get_train_loader()
    # val_loader = chexpert.get_val_loader()
    test_loader = chexpert.get_test_loader()

    classifier = StableDiffusionClassifier(config, label_to_text_mapper=l_to_t_mapper)

    metric_output, _, _ = classifier.inference(test_loader, metrics)
    try:
        print([{k: round(v.item(), 4) for k, v in d.items()} for d in metric_output])
    except:
        print(metric_output)
        # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
