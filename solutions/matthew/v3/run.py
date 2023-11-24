import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
import copy
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cgiar.data import CGIARDataset
from cgiar.matthew_models.densenet import DenseNet_Custom
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
    OUTPUT_DIR=get_dir('solutions/matthew/v3', args.subfolder)
    
    print(f"Saving things to {OUTPUT_DIR}")

    # ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.2),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.2),    # Random vertical flip
    transforms.RandomRotation(degrees=15),    # Random rotation between -15 to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Minor affine shifts
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),  # Adding Gaussian noise
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the regression model
    # model = Resnet50_V1()
    model = DenseNet_Custom()
    model = model.to(device)

    # Define loss function (mean squared error) and optimizer (e.g., Adam)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()

    losses = []  # List to store loss values
    val_losses = []

    # KFold cross-validator
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    best_model_wts = None
    lowest_loss = float('inf')
    patience = 3  # Number of epochs with no improvement after which training will be stopped
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f'Epoch [{epoch+1}/{EPOCHS}]')
        epoch_loss = 0.0
        epoch_val_loss = 0.0

        # Loop through folds
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
            print(f'  FOLD {fold}')
            print('  --------------------------------')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            train_loader = DataLoader(
                train_dataset, 
                batch_size=TRAIN_BATCH_SIZE, 
                sampler=train_subsampler
            )
            val_loader = DataLoader(
                train_dataset, 
                batch_size=TEST_BATCH_SIZE, 
                sampler=val_subsampler
            )

            # Training loop
            model.train()
            for _, images, damages in train_loader:
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs, damages.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation loop
            model.eval()
            with torch.no_grad():
                for _, images, damages in val_loader:
                    outputs = model(images.to(device))
                    loss = criterion(outputs, damages.to(device))
                    epoch_val_loss += loss.item()

        # Calculate average loss over all folds
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        avg_epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
        print(f'\t Train Loss {avg_epoch_loss}')
        print(f'\t Validation Loss {avg_epoch_val_loss}')
        val_losses.append(avg_epoch_val_loss)
        losses.append(avg_epoch_loss)

        # Early stopping logic
        if avg_epoch_val_loss < lowest_loss:
            lowest_loss = avg_epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
        
    # Plot the loss curve
    try:
        plt.plot(range(1, len(losses)+1), losses)
        # plt.plot(range(1, len(losses)+1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / 'train_loss.png')
    except:
        print("issue with image generation")
    
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