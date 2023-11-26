import timm
import torch.nn as nn


class Resnet50_V1(nn.Module):
    def __init__(self):
        super(Resnet50_V1, self).__init__()
        
        self.model = timm.create_model(
            'resnet50.a1_in1k',
            pretrained=True
        )
        
        # Modify the final classification layer 
        # to output a single regression value
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.model(x)
    
class TIMM_Model(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(TIMM_Model, self).__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained
        )
        
        if any([model_name.startswith(n) for n in ['efficientnet']]):
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        elif any([model_name.startswith(n) for n in ['convnext', 'regnet']]):
            num_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        elif any([model_name.startswith(n) for n in ['vit', 'deit']]):  
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        elif any([model_name.startswith(n) for n in ['eva']]):
            self.model.head = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        elif any([model_name.startswith(n) for n in ['resn']]):
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, 5),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)