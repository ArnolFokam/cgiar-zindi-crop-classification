import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class Resnet50_V1(nn.Module):
    def __init__(self):
        super(Resnet50_V1, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x
    
class Resnet50_V2(nn.Module):
    def __init__(self):
        super(Resnet50_V2, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 101)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x
    
class Resnet50_V3(nn.Module):
    def __init__(self):
        super(Resnet50_V3, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x
    
class XCITMultipleMLP(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 num_mlps: int = 1,
                 hidden_size: int = 128) -> None:
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.model.head.in_features
        
        # modify head
        self.model.head_drop = nn.Dropout(0.2)
        self.model.head = nn.Sequential(
            nn.Linear(
                num_features, 
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, num_mlps),
            nn.ReLU(),
        )
        self.num_mlps = num_mlps
        
    def forward(self, x):
        # Forward pass through the CNN and regression layers
        # Get the MLP predictions for a particular growth stage
        growth_stage, _, image = x
        mask = F.one_hot(growth_stage, num_classes=self.num_mlps)
        predictions = self.model(image) * mask.float()
        return predictions.sum(dim=-1)
    
class ResNetMultipleMLP(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 num_mlps: int = 1,
                 hidden_size: int = 128) -> None:
        super().__init__()
        
        # Load a pretrained CNN (e.g., ResNet50) with weights
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(
                num_features, 
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, num_mlps),
            nn.ReLU(),
        )
        
        # modify head
        self.num_mlps = num_mlps
        
    def forward(self, x):
        # Forward pass through the CNN and regression layers
        # Get the MLP predictions for a particular growth stage
        growth_stage, _, image = x
        mask = F.one_hot(growth_stage, num_classes=self.num_mlps)
        predictions = self.model(image) * mask.float()
        return predictions.sum(dim=-1)
        