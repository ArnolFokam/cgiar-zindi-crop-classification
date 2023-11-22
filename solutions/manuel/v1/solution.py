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


if __name__ == "__main__":
    # Define hyperparameters
    SEED=42
    LR=1e-4
    EPOCHS=20
    IMAGE_SIZE=224
    TRAIN_BATCH_SIZE=64
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create instances of CGIARDataset for training and testing
    train_dataset = CGIARDataset(root_dir=DATA_DIR, split='train', transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

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
        
            for _, images, extents in train_loader:
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs.squeeze(), extents.to(device).squeeze())
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
    
    train_predictions = []
    
    # get and save the train predictions
    with torch.no_grad():
        for ids, images, _ in train_loader:
            outputs = model(images.to(device))
            outputs = outputs.squeeze().tolist()
            train_predictions.extend(list(zip(ids, outputs)))
    
    train_dataset.df['predicted_extent'] = train_dataset.df['ID'].map(dict(train_predictions))
    train_dataset.df.to_csv(OUTPUT_DIR / 'train_predictions.csv', index=False)

    test_dataset = CGIARDataset(root_dir=DATA_DIR, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    predictions = []

    with torch.no_grad():
        for ids, images, _ in test_loader:
            outputs = model(images.to(device))
            outputs = outputs.squeeze().tolist()
            predictions.extend(list(zip(ids, outputs)))

    # load the sample submission file and update the extent column with the predictions
    submission_df = pd.read_csv('data/SampleSubmission.csv')

    # update the extent column with the predictions
    submission_df['extent'] = submission_df['ID'].map(dict(predictions))

    # save the submission file and trained model
    submission_df.to_csv(OUTPUT_DIR / 'submission.csv', index=False)