import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.wavelet import wavelet_dec_2

class CheXpertDataset(Dataset):
    def __init__(self, data_path, split="train", wavelet_transform=False, image_size=(256,256), augment=False, sens_attr=None):
        """
        Args:
            data_path (str): Path to the CheXpert dataset.
            split (str): Dataset split to use. One of "train", "valid", "test".
            wavelet_transform (bool): Whether to apply wavelet transform to the images.
            image_size (tuple): Size to resize the images to.
            augment (bool): Whether to apply data augmentation to the images.

        Note:
            Everything is derived from the train split.
        """
        self.wavelet_transform = wavelet_transform
        self.data_path = data_path
        self.split = split if split != "test" else "valid"
        self.image_size = image_size
        self.augment = augment

        self.sens_attr = sens_attr

        # Get images path
        self.img_dir = os.path.join(self.data_path, "train")

        # Check the dataset split and apply filtering accordingly
        if split == "train": # Take first 80% of the data
            # self.data = pl.read_csv("dataset/splits/chexpert-train.csv")
            self.data = pl.read_csv("dataset/splits/chexpert-train-with-metadata.csv")

        elif split == "valid": # Take first half of last 20%) of the data
            # self.data = pl.read_csv("dataset/splits/chexpert-valid.csv")
            self.data = pl.read_csv("dataset/splits/chexpert-valid-with-metadata.csv")

        elif split == "test": # Take second half of last 20% of the data
            self.data = pl.read_csv("dataset/splits/chexpert-test1.csv")
            # self.data = pl.read_csv("dataset/splits/chexpert-test-with-metadata.csv")

        # Print length of dataset
        print(f"Dataset length: {len(self.data)}")

        # Print label column name
        print(f"Label column name: {self.data.columns[1]}")
        
        # Get unique label and their counts from the polars dataframe
        print(self.data.group_by("Pleural Effusion").count().sort("Pleural Effusion"))

    def transforms(self):
        if not self.augment or self.split != "train":
            return transforms.Compose([
                transforms.Resize((self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        else: # if self.augment and self.split == "train":
            return transforms.Compose([
                transforms.RandomRotation(degrees=30),  # Random rotation
                transforms.RandomHorizontalFlip(p=0.5),  # Equivalent to RandFlip (spatial_axis=0)
                transforms.RandomVerticalFlip(p=0.5),  # Equivalent to RandFlip (spatial_axis=1)
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), 
                transforms.Resize((self.image_size)), 
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    def _filter_study1_frontal(self):
        # Load CSV
        df = pl.read_csv(self.csv_path)

        # Filter for Study1 frontal images
        study1_frontal = df.filter(pl.col("Path").str.contains("study1/view1_frontal.jpg"))

        # Keep only relevant columns (Path and Pleural Effusion label)
        study1_frontal = study1_frontal.select(["Path", "Pleural Effusion", "No Finding"])

        # Replace NaN labels with 0
        study1_frontal = study1_frontal.with_columns(
            pl.col("Pleural Effusion").fill_null(0),
            pl.col("No Finding").fill_null(0)
        )

        # Drop rows where label is -1
        study1_frontal = study1_frontal.filter(
            pl.col("Pleural Effusion") != -1,
            pl.col("No Finding") != -1    
        )

        # Create new column that is XOR of Pleural Effusion and No Finding
        study1_frontal = study1_frontal.with_columns(
            ((pl.col("Pleural Effusion")>0) ^ (pl.col("No Finding")>0)).alias("healthy_or_sick")
        )

        # Drop rows where healthy_or_sick is 0
        study1_frontal = study1_frontal.filter(pl.col("healthy_or_sick") == 1)

        # Separate the active and inactive labels
        active_df = study1_frontal.filter(pl.col("Pleural Effusion") == 1)
        inactive_df = study1_frontal.filter(pl.col("Pleural Effusion") == 0)

        # Take the minimum count of the two labels
        min_count = min(active_df.height, inactive_df.height)

        # Sample the data to have equal number of active and inactive labels
        active_df = active_df.sample(n=min_count, with_replacement=False, seed=42)
        inactive_df = inactive_df.sample(n=min_count, with_replacement=False, seed=42)

        # Concatenate the two dataframes
        study1_frontal = pl.concat([active_df, inactive_df])

        # Shuffle the dataframe
        study1_frontal = study1_frontal.sample(n=len(study1_frontal), shuffle=True, seed=42)

        return study1_frontal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        # rel_path = row["Path"].item().split("/")[1:]
        # rel_path = os.path.join(*rel_path)
        # img_path = os.path.join(self.data_path, rel_path)
        img_path = os.path.join(self.data_path, row["Path"].item())
        label = int(row["Pleural Effusion"].item())

        try:
            sens_attr = row[self.sens_attr].item()
        except:
            sens_attr = None
        
        try:
            prompt_with_metadata = row["prompt_with_metadata"].item()
        except:
            prompt_with_metadata = None

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        image = self.transforms()(image)

        if self.wavelet_transform:
            image = wavelet_dec_2(image) / 2

        return image, label, prompt_with_metadata, sens_attr
    
class CheXpertDataLoader:
    def __init__(self, wavelet_transform, data_path, batch_size=64, num_workers=4, image_size=(256,256), cf_label=None, augment=False, sens_attr=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cf_label = cf_label
        
        self.sens_attr = sens_attr

        self.train_dataset = CheXpertDataset(data_path=self.data_path, split="train", image_size=image_size, wavelet_transform=wavelet_transform, augment=augment, sens_attr=sens_attr)
        self.val_dataset = CheXpertDataset(data_path=self.data_path, split="valid", image_size=image_size, wavelet_transform=wavelet_transform, augment=augment, sens_attr=sens_attr)
        self.test_dataset = CheXpertDataset(data_path=self.data_path, split="test", image_size=image_size, wavelet_transform=wavelet_transform, augment=augment, sens_attr=sens_attr)

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
        images, labels, prompt_with_metadata, sens_attr = zip(*batch)

        if self.cf_label is not None:
            # Make all labels the cf_label
            labels = [self.cf_label for _ in labels]

        # print("SENS ATTR: ", sens_attr)
        
        if(self.sens_attr is None):
            return {
                "images": torch.stack(images),
                "prompt": torch.tensor(labels),
                "prompt_with_metadata": prompt_with_metadata
            }
        else:
            return {
                "images": torch.stack(images),
                "prompt": torch.tensor(labels),
                "prompt_with_metadata": prompt_with_metadata,
                "sens_attr": torch.tensor(sens_attr)
            }

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader