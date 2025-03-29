import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.wavelet import wavelet_dec_2

class ISICDataset(Dataset):
    def __init__(self, data_path, split="train", wavelet_transform=False, image_size=(256,256)):
        """
        Args:
            data_path (str): Path to the ISIC dataset.
            split (str): Dataset split to use. One of "train", "valid", "test".
            wavelet_transform (bool): Whether to apply wavelet transform to the images.
            image_size (tuple): Size to resize the images to.

        Note:
            Everything is derived from the train split.
        """
        self.wavelet_transform = wavelet_transform
        self.data_path = data_path
        self.split = split if split != "test" else "valid"
        self.image_size = image_size

        # Get images path
        self.img_dir = os.path.join(self.data_path, "train")

        # Check the dataset split and apply filtering accordingly
        if split == "train": # Take first 80% of the data
            self.data = pl.read_csv("dataset/splits/isic-train.csv")
        elif split == "valid": # Take first half of last 20%) of the data
            self.data = pl.read_csv("dataset/splits/isic-valid.csv")
        elif split == "test": # Take second half of last 20% of the data
            self.data = pl.read_csv("dataset/splits/isic-test.csv")

        # Print length of dataset
        print(f"Dataset length: {len(self.data)}")

        # Print label column name
        print(f"Label column name: {self.data.columns[1]}")
        
        # Get unique label and their counts from the polars dataframe
        print(self.data.group_by("target").count().sort("target"))

    def transforms(self):
        return transforms.Compose([
            transforms.Resize((self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        rel_path = f'images/{row["image_name"].item()}.jpg'
        img_path = os.path.join(self.data_path, rel_path)
        label = int(row["target"].item())

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        image = self.transforms()(image)

        if self.wavelet_transform:
            image = wavelet_dec_2(image) / 2

        return image, label
    
class ISICDataLoader:
    def __init__(self, wavelet_transform, data_path, cf_label=None, batch_size=64, num_workers=4, image_size=(256,256)):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cf_label = cf_label

        # Initialize datasets
        self.train_dataset = ISICDataset(data_path=self.data_path, split="train", wavelet_transform=wavelet_transform, image_size=image_size)
        self.val_dataset = ISICDataset(data_path=self.data_path, split="valid", wavelet_transform=wavelet_transform, image_size=image_size)
        self.test_dataset = ISICDataset(data_path=self.data_path, split="test", wavelet_transform=wavelet_transform, image_size=image_size)

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def collate_fn(self, batch):
        # Extract the images and labels from the batch
        images, labels = zip(*batch)

        if self.cf_label is not None:
            # Make all labels the cf_label
            original_labels = labels
            labels = [self.cf_label for _ in labels]

        return {
            "images": torch.stack(images),
            "prompt": torch.tensor(labels),
            "original_labels": torch.tensor(original_labels) if self.cf_label is not None else None,
        }

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
