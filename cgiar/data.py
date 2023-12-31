import pathlib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import torch
import torchvision.transforms.functional as F
import torch.nn as nn
from tqdm import tqdm

from cgiar.utils import get_dir


def resize(image: Image, size: int):
    # Calculate the aspect ratio of the input image
    width, height = image.size
    aspect_ratio = width / height

    # Determine the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_height = size
        new_width = int(size * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


class RandomErasing(transforms.RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p, scale, ratio, value, inplace)
        
    def __call__(self, img: Image.Image):
        img = transforms.ToTensor()(img)
        img = super().__call__(img)
        img = transforms.ToPILImage()(img)
        return img
        

augmentations = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.5),
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=0.5),
    "RandomRotation": transforms.RandomRotation(degrees=45),
    "ColorJitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    "RandomAffine": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "RandomPerspective": transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    "RandomErasing": RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    "RandomGrayscale": transforms.RandomGrayscale(p=0.2),
    "RandomAffineWithResize": transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), interpolation=Image.BILINEAR),
    "RandomPosterize": transforms.RandomPosterize(bits=4),
    "RandomSolarize": transforms.RandomSolarize(threshold=128),
    "RandomEqualize": transforms.RandomEqualize(p=0.1),
    "RandomBlur": transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    "Identity": nn.Identity(),
}

class CGIARDataset(Dataset):
    """Pytorch data class"""
    
    # get the csv file name from the split
    split_to_csv_filename = {
        "train": "Train",
        "test": "Test"
    }
    
    columns = ["DR", "G", "ND", "WD", "other"]
    
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None,
                 initial_image_size : int = 512):
        """
        Args:
            root_dir (pathlib.Path): Root directory containing all the image files.
            split (string): Split name ('train', 'test', etc.) to determine the CSV file.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.images_dir = get_dir(root_dir) / "images"
        self.transform = transform
        self.split = split

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')
        
        self.images = {}
        
        # Load all the images into memory
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_path = self.images_dir / row['filename']
            image = Image.open(image_path)
            image = resize(image, initial_image_size)
            self.images[idx] = image
            
        # Concatenate the one-hot encoded 
        # DataFrame with the original DataFrame
        if self.split == "train":
            self.df = pd.concat([
                self.df,
                pd.get_dummies(self.df['damage'])
            ], axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self._transform_image(image)
        
        damage = [-1]
        if self.split == "train":
            damage = self.df.iloc[idx, self.df.columns.get_indexer(self.columns)]
        
        damage = torch.FloatTensor(damage)
        return self.df['ID'].iat[idx], image, damage
    
    def _transform_image(self, image):
        return self.transform(image)


class CGIARDatasetV2(Dataset):
    """Pytorch data class"""
    
    # get the csv file name from the split
    split_to_csv_filename = {
        "train": "Train",
        "test": "Test"
    }
    
    columns = ["DR", "G", "ND", "WD", "other"]
    
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None,
                 initial_image_size : int = 512):
        """
        Args:
            root_dir (pathlib.Path): Root directory containing all the image files.
            split (string): Split name ('train', 'test', etc.) to determine the CSV file.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.images_dir = get_dir(root_dir) / "images"
        self.transform = transform
        self.split = split

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')
        
        self.images = {}
        
        # Load all the images into memory
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_path = self.images_dir / row['filename']
            image = Image.open(image_path)
            image = resize(image, initial_image_size)
            self.images[idx] = image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self._transform_image(image)
        
        damage = -1
        if self.split == "train":
            damage = self.columns.index(self.df.iloc[idx]['damage'])
        
        return self.df['ID'].iat[idx], image, damage
    
    def _transform_image(self, image):
        return self.transform(image)