import pathlib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import torch
import cv2
import random
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
    
    columns = ["ID", "filename", "growth_stage", "extent", "season"]
    
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None,
                 initial_size : int =512):
        """
        Args:
            root_dir (pathlib.Path): Root directory containing all the image files.
            split (string): Split name ('train', 'test', etc.) to determine the CSV file.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.images_dir = get_dir(root_dir) / split
        self.transform = transform
        self.split = split

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')
        # self.df = self.df.iloc[:50, :]
        
        
        self.images = {}
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_path = self.images_dir / row['filename']
            image = Image.open(image_path)
            image = resize(image, initial_size)
            self.images[idx] = image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self._transform_image(image)
        
        extent = -1
        if self.split == "train":
            extent = self.df.iloc[idx, self.columns.index("extent")]
        
        extent = torch.FloatTensor([extent])
        return self.df.iloc[idx, self.columns.index("ID")], image, extent
    
    def _transform_image(self, image):
        return self.transform(image)
        
        
class CGIARDataset_V2(CGIARDataset):
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None, 
                 initial_size: int = 512, 
                 num_views=1):
        super().__init__(root_dir, split, transform, initial_size)
        self.num_views = num_views
        
    def _transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_views)]
    
class CGIARDataset_V3(CGIARDataset_V2):
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str = 'train', 
                 transform=None, 
                 initial_size: int = 512, 
                 num_views=1):
        super().__init__(root_dir, split, transform, initial_size, num_views)
        
        if self.split == "train":
            # unique values from the "extent" 
            # column and use them as classes
            self.classes = self.df["extent"].unique()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
            
            # get classes weights
            class_counts = self.df["extent"].value_counts().sort_index()
            total_samples = len(self.df)
            self.class_weights = total_samples / (class_counts * len(class_counts))
            self.class_weights = self.class_weights.to_numpy()
            
    def __getitem__(self, idx):
        index, image, extent = super().__getitem__(idx)
        
        if extent.item() != -1:
            extent = self.class_to_idx[extent.item()]
            extent = torch.LongTensor([extent])
            
        return index, image,  extent
    
class CGIARDataset_V4(Dataset):
    growth_stage_to_class = {
        'S': 0,
        'V': 1,
        'F': 2,
        'M': 3
    }
    
    season_to_class = {
        'SR2021': 0,
        'LR2021': 1,
        'LR2020': 2,
        'SR2020': 3,
    }
    
    columns = ["ID", "filename", "growth_stage", "season"]
    
    def __init__(self, 
                 features, 
                 images,
                 num_views=1,
                 labels=None,
                 transform=None,
                 *args,
                 **kwargs):
        # save data
        self.features = features
        self.labels = labels
        self.images = images
        
        # transforms
        self.num_views=num_views
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        index = self.features.iloc[idx, self.columns.index("ID")]
        growth_stage = self.features.iloc[idx, self.columns.index("growth_stage")]
        season = self.features.iloc[idx, self.columns.index("season")]
        image = self.images[index]
        
        extent = -1
        if self.labels is not None:
            extent = self.labels.iloc[idx]
            
        if self.transform:
            images = self._transform_image(image)
        
        return (
            index, 
            images, 
            torch.LongTensor([self.growth_stage_to_class[growth_stage]]), 
            torch.LongTensor([self.season_to_class[season]]), 
            torch.LongTensor([extent])
        )
        
    def _transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_views)]
    
    @staticmethod
    def load_images(dataframe, folder, initial_size):
        images = {}
        
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            index = row['ID']
            image_path = folder / row['filename']
            image = Image.open(image_path)
            image = resize(image, initial_size)
            images[idx] = (index, image)
            
        return images
        
        