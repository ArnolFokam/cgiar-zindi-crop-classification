from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import timm
import argparse

from cgiar.data import CGIARDataset_V2, augmentations
from cgiar.utils import get_dir

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Image Augmentation Program")
    parser.add_argument("--model_name", type=str, help="Specify the pre-trained model to the get the representations from")
    parser.add_argument("--index", default="./", type=str, help="Specify the index of the run")
    args = parser.parse_args()
    
    # Define hyperparameters
    SEED=42
    IMAGE_SIZE=224
    TRAIN_BATCH_SIZE=64
    TEST_BATCH_SIZE=32
    NUM_VIEWS=10

    DATA_DIR=get_dir('data')
    OUTPUT_DIR=get_dir('solutions/v8', args.index)

    # ensure reproducibility
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(args.model_name, args.index)
        
    # Define transform for image preprocessing
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        augmentations["RandomEqualize"],
        augmentations["RandomAffine"],
        augmentations["RandomErasing"],
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create instances of CGIARDataset for training and testing
    train_dataset = CGIARDataset_V2(root_dir=DATA_DIR, split='train', transform=transform, num_views=NUM_VIEWS)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    # Initialize the regression model
    model = timm.create_model(args.model_name, pretrained=True)
    model.to(device)
    model.eval()

    train_representations = []
    train_targets = []
    
    with torch.no_grad():
        for _, images_list, extents in train_loader:
            representations = torch.stack([model(images.to(device)) for images in images_list]).cpu().numpy()
            train_representations.extend(representations)
            train_targets.extend(extents.repeat(NUM_VIEWS, 1).numpy())
        
    torch.cuda.empty_cache()
    
    train_representations = np.concatenate(train_representations)
    train_targets = np.concatenate(train_targets)
    
    knn_regressor = KNeighborsRegressor(n_jobs=-1, n_neighbors=10)
    knn_regressor.fit(train_representations, train_targets)
    
    train_loader.dataset.num_views = NUM_VIEWS
    train_predictions = []
    train_targets = []
    
    # get and save the train predictions
    with torch.no_grad():
        for ids, images_list, extents in train_loader:
            # get the mode argmax predictions from each model
            outputs = torch.stack([model(images.to(device)) for images in images_list]).mean(dim=0).cpu().numpy()
            outputs = knn_regressor.predict(outputs).tolist()
            train_predictions.extend(list(zip(ids, outputs)))
            train_targets.append(extents.numpy())
            
    torch.cuda.empty_cache()
    train_targets = np.concatenate(train_targets)
            
    mse = mean_squared_error(train_targets, [output[1] for output in train_predictions])
    print("Mean Squared Error:", mse ** 0.5)
            
    train_dataset.df['predicted_extent'] = train_dataset.df['ID'].map(dict(train_predictions))
    train_dataset.df.to_csv(OUTPUT_DIR / 'train_predictions.csv', index=False)
    
    torch.cuda.empty_cache()
    
    test_dataset = CGIARDataset_V2(root_dir=DATA_DIR, split='test', num_views=NUM_VIEWS, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    predictions = []

    with torch.no_grad():
        for ids, images_list, _ in test_loader:
            # average predictions from all the views
            outputs = torch.stack([model(images.to(device)) for images in images_list]).mean(dim=0).cpu().numpy()
            outputs = knn_regressor.predict(outputs).tolist()
            predictions.extend(list(zip(ids, outputs)))

    # load the sample submission file and update the extent column with the predictions
    submission_df = pd.read_csv('data/SampleSubmission.csv')

    # update the extent column with the predictions
    submission_df['extent'] = submission_df['ID'].map(dict(predictions))

    # save the submission file and trained model
    submission_df.to_csv(OUTPUT_DIR / 'submission.csv', index=False)