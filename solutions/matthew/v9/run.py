import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import random_split
from torch.utils.data import Subset

from cgiar.data import CGIARDataset
from cgiar.matthew_models.efficientnet_v2 import EfficientNetB4_Custom
from cgiar.utils import get_dir, time_activity


def parse_args():
    parser = argparse.ArgumentParser(description='CGIAR Model Training and Evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--initial_image_size', type=int, default=512, help='Initial image size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for training')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--subfolder', type=str, default="#1", help='Index of the learning rate')

    args = parser.parse_args()
    return args

def run():
    args = parse_args()
    
    # Use args instead of hard-coded values
    SEED = args.seed
    LR = args.lr
    EPOCHS = args.epochs
    INITIAL_IMAGE_SIZE = args.initial_image_size
    IMAGE_SIZE = args.image_size
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size

    DATA_DIR=get_dir('data')
    OUTPUT_DIR=get_dir('solutions/matthew/v9', args.subfolder)
    
    print(f"Saving things to {OUTPUT_DIR}")

    # ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Define transform for image preprocessing
    # Extra transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])

    # Create instances of CGIARDataset for training and testing
    train_dataset = CGIARDataset(
        split='train', 
        root_dir=DATA_DIR, 
        transform=transform,
        initial_image_size=INITIAL_IMAGE_SIZE,
    )

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation sets
    train_loader = DataLoader(
        train_subset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=TEST_BATCH_SIZE, 
        shuffle=False
    )

    # Initialize the regression model
    model = EfficientNetB4_Custom()
    model = model.to(device)

    # Define loss function (mean squared error) and optimizer (e.g., Adam)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()

    losses = []  # List to store loss values
    val_losses = []

        # Training loop for regression model
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_val_loss = 0.0

        with time_activity(f'Epoch [{epoch+1}/{EPOCHS}]'):
            # Training
            model.train()
            best_model = copy.deepcopy(model)
            for _, images, damages in train_loader:
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs, damages.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            losses.append(avg_epoch_loss)
            # Validation
            model.eval()
            with torch.no_grad():
                for _, images, damages in val_loader:
                    outputs = model(images.to(device))
                    loss = criterion(outputs, damages.to(device))
                    epoch_val_loss += loss.item()

            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_epoch_val_loss)
            if len(val_losses) > 1 and val_losses[-1] > val_losses[-2]:
                break

            print(f'Train Loss: {avg_epoch_loss}')
            print(f'Validation Loss: {avg_epoch_val_loss}')
    
    model = best_model
    model.eval()
    torch.save(model.state_dict(), OUTPUT_DIR / 'model.pt')
        
    # Plot the loss curve
    plt.plot(range(1, len(losses)+1), losses, label='train')
    plt.plot(range(1, len(losses)+1), val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'train_loss.png')
    
    # evaluation
    model.eval()

    # make predictions on the test set
    test_dataset = CGIARDataset(root_dir=DATA_DIR, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    predictions = []

    with torch.no_grad():
        for ids, images, _ in test_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            outputs = outputs.tolist()
            predictions.extend(list(zip(ids, outputs)))
            
    test_predictions = defaultdict(lambda: [0, 0, 0, 0, 0])
    test_predictions.update(dict(predictions))

    # load the sample submission file and update the extent column with the predictions
    submission_df = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

    # update the extent column with the predictions
    submission_df.loc[:, test_dataset.columns] = submission_df['ID'].map(test_predictions).to_list()

    # save the submission file and trained model
    submission_df.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
