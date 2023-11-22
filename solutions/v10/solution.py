import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


from cgiar.model import XCITMultipleMLP
from cgiar.utils import get_dir, time_activity
from cgiar.data import CGIARDataset_V4, augmentations


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Image Augmentation Program")
    parser.add_argument("--model_name", type=str, help="Specify the pre-trained model to the get the representations from")
    parser.add_argument("--index", default="./", type=str, help="Specify the index of the run")
    args = parser.parse_args()
    
    # Define hyperparameters
    SEED=42
    LR=1e-4
    EPOCHS=35
    IMAGE_SIZE=224
    INITIAL_SIZE=512
    TRAIN_BATCH_SIZE=128
    TEST_BATCH_SIZE=64
    HIDDEN_SIZE=128
    NUM_FOLDS=5
    NUM_VIEWS=20

    DATA_DIR=get_dir('data')
    OUTPUT_DIR=get_dir('solutions/v10', args.index)

    # ensure reproducibility
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(args.model_name, args.index)
        
    # Define transform for image preprocessing
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        augmentations["RandomEqualize"],
        augmentations["RandomBlur"],
        augmentations["RandomErasing"],
        augmentations["RandomAffine"],
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = nn.Identity()
    
    test_transform = transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        augmentations["RandomEqualize"],
        augmentations["RandomAffine"],
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data frame from csv
    df_train = pd.read_csv(DATA_DIR / 'Train.csv')
    X_train = df_train.drop(columns=['extent'], axis=1)
    y_train = df_train['extent']
    
    train_images = CGIARDataset_V4.load_images(X_train, DATA_DIR / "train", INITIAL_SIZE)
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Perform stratified k-fold cross-validation
    fold_idx = 0
    
    with time_activity(f'Strain K-Fold Cross-Validation [{NUM_FOLDS} folds]'):
        models = {}
        fold_losses = {}
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            
            # Get the train and val data for this fold
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            train_images_fold = dict([train_images[idx] for idx in train_idx])
            
            # ==== TRAINING LOOP ==== #
            
            # Create DataLoader instances
            train_dataset_fold = CGIARDataset_V4(
                transform=transform,
                labels=y_train_fold,
                features=X_train_fold,
                images=train_images_fold,
                target_transform=target_transform,
            )
            
            # Create DataLoader instances
            train_loader_fold = DataLoader(
                train_dataset_fold, 
                batch_size=TRAIN_BATCH_SIZE, 
                shuffle=True
            )
            
            # Initialize the regression model
            model = XCITMultipleMLP(
                model_name=args.model_name,
                pretrained=True,
                num_mlps=len(train_loader_fold.dataset.growth_stage_to_class),
                hidden_size=HIDDEN_SIZE
            )
            model.to(device)
            
            # # Define loss function (mean squared error) and optimizer (e.g., Adam)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            model.train()
            
            train_losses = []  # List to store loss values

            # Training loop for regression model
            with time_activity(f'Training on Fold [{fold_idx + 1}/{NUM_FOLDS}]'):
                
                for epoch in range(EPOCHS):
                    
                    epoch_loss = 0.0
                    
                    with time_activity(f'Epoch [{epoch+1}/{EPOCHS}]'):
                    
                        for _, images_list, growth_stage, season, extents in train_loader_fold:
                            images = images_list[0]
                            optimizer.zero_grad()
                            outputs = model((
                                growth_stage.to(device).squeeze(),
                                season.to(device).squeeze(),
                                images.to(device)
                            ))
                            loss = criterion(
                                outputs.squeeze(), 
                                extents.to(device).squeeze().float()
                            )
                            loss.backward()
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            
                        # Calculate average epoch loss
                        avg_epoch_loss = epoch_loss / len(train_loader_fold)
                        train_losses.append(avg_epoch_loss)
                    
                        print(f'Train Loss: {avg_epoch_loss}')
            
                # Plot the loss curve
                plt.plot(range(1, EPOCHS+1), train_losses)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{fold_idx} Training Loss')
                plt.grid(True)
                plt.savefig(OUTPUT_DIR / f'train_loss_fold_{fold_idx}.png')
                plt.clf()
            
            torch.cuda.empty_cache()
            
            # ==== VALIDATION LOOP ==== #
            
            with time_activity(f'Evaluating on Fold [{fold_idx + 1}/{NUM_FOLDS}]'):
            
                model.eval()
                
                val_images_fold = dict([train_images[idx] for idx in val_idx])
                
                # Create DataLoader instances
                val_dataset_fold = CGIARDataset_V4(
                    transform=test_transform,
                    labels=y_val_fold,
                    features=X_val_fold,
                    num_views=NUM_VIEWS,
                    images=val_images_fold,
                    target_transform=target_transform,
                )
                
                # Create DataLoader instances
                val_loader_fold = DataLoader(
                    val_dataset_fold, 
                    batch_size=TEST_BATCH_SIZE, 
                    shuffle=False
                )
                
                val_loss = 0.0  # List to store loss values
        
                # Evaluate the model on the val data
                with torch.no_grad():
                    for _, images_list, growth_stage, season, extents in val_loader_fold:
                        # average predictions from all the views
                        outputs = torch.stack([model((
                            growth_stage.to(device).squeeze(),
                            season.to(device).squeeze(),
                            images.to(device)
                        )) for images in images_list]).mean(dim=0)
                        loss = criterion(
                            outputs.squeeze(), 
                            extents.to(device).squeeze().float()
                        )
                        
                        val_loss += loss.item()
                        
                # Calculate average val loss
                avg_val_loss = (val_loss / len(val_loader_fold)) ** 0.5
                print(f"Val Loss = {avg_val_loss}")
                fold_losses[fold_idx] = avg_val_loss
                
            # save model
            torch.save(model.state_dict(), OUTPUT_DIR / f'model_fold_{fold_idx}.pth')
            models[fold_idx] = model
            models[fold_idx].cpu()
            
            fold_idx += 1
                
        # plot bar of the fold losses
        plt.bar(fold_losses.keys(), fold_losses.values())
        plt.xlabel('Fold')
        plt.ylabel('Val Loss')
        plt.title(f'{fold_idx} Validation Loss')
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / f'val_loss_folds.png')
        plt.clf()
        
        # save fold loss as json
        pd.Series(fold_losses).to_json(OUTPUT_DIR / 'val_loss_folds.json')
    