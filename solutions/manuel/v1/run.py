from collections import defaultdict
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cgiar.data import CGIARDataset
from cgiar.model import Resnet50_V1
from cgiar.utils import get_dir, time_activity


def run():
    
    # Define hyperparameters
    SEED=42
    LR=1e-4
    EPOCHS=40
    INITIAL_IMAGE_SIZE=512
    IMAGE_SIZE=224
    TRAIN_BATCH_SIZE=128
    TEST_BATCH_SIZE=32

    DATA_DIR=get_dir('data')
    OUTPUT_DIR=get_dir('solutions/manuel/v1')

    # ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Define transform for image preprocessing
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create instances of CGIARDataset for training and testing
    train_dataset = CGIARDataset(
        split='train', 
        root_dir=DATA_DIR, 
        transform=transform,
        initial_image_size=INITIAL_IMAGE_SIZE,
    )

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True
    )

    # Initialize the regression model
    model = Resnet50_V1()
    model = model.to(device)

    # Define loss function (mean squared error) and optimizer (e.g., Adam)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()

    losses = []  # List to store loss values

    # Training loop for regression model
    for epoch in range(EPOCHS):
        
        epoch_loss = 0.0
        
        with time_activity(f'Epoch [{epoch+1}/{EPOCHS}]'):
        
            for _, images, damages in train_loader:
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs, damages.to(device))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            losses.append(avg_epoch_loss)
        
            print(f'Loss: {avg_epoch_loss}')
    
    torch.save(model.state_dict(), OUTPUT_DIR / 'model.pt')
        
    # Plot the loss curve
    plt.plot(range(1, EPOCHS+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
