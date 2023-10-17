import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.optim as optim
from Utils import Utils


class MultiLabelModel(torch.nn.Module):
    def __init__(self, num_classes, weights=None):
        super(MultiLabelModel, self).__init__()
        self.model = resnet50(weights=weights)  
        
        num_features = self.model.fc.in_features  
        self.model.fc = torch.nn.Linear(num_features, num_classes)  

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)



if __name__ == '__main__':
    torch.manual_seed(0)
    dir = './EuroSAT_RGB/EuroSAT_RGB/'
    util = Utils(dir)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = util.split_data(multilabel=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLabelModel(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT).to(device)
    
        