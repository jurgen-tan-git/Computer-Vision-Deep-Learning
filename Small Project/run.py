import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch

class CustomImageDataset(Dataset):
    def __init__(self, dir, rootpath, transform=None, test_size=0.2, val_size=0.1, random_state=51):
        self.transform = transform
        self.dataset = {}
        
        for folder in dir:
            for file in os.listdir(rootpath + folder):
                self.dataset[file] = folder

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            list(self.dataset.keys()), list(self.dataset.values()), test_size=test_size, random_state=random_state
        )
        # Split the train data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=val_size, random_state=random_state
        )
        # Binarize labels using LabelBinarizer
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(self.y_train)
        self.num_classes = len(self.label_binarizer.classes_)

    def get_train_data(self):
        return self.X_train, self.label_binarizer.transform(self.y_train)

    def get_val_data(self):
        return self.X_val, self.label_binarizer.transform(self.y_val)

    def get_test_data(self):
        return self.X_test, self.label_binarizer.transform(self.y_test)

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=102, weights=None):
        super(ResNet, self).__init__()
        self.model = resnet18(weights=weights)  # Load the pre-trained ResNet-50 model

        # Modify the final fully connected layer (fc) for your custom classification task
        num_features = self.model.fc.in_features  # Get the number of input features to the final layer
        self.model.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

def createDataLoader(dataset, batch_size):
    train_ds = DataLoader(dataset.get_train_data(), batch_size=batch_size, shuffle=True)
    val_ds = DataLoader(dataset.get_val_data(), batch_size=batch_size, shuffle=True)
    test_ds = DataLoader(dataset.get_test_data(), batch_size=batch_size, shuffle=True)
    return {'train': train_ds, 'val': val_ds, 'test': test_ds}

    
if __name__ == '__main__':
    ds = CustomImageDataset(dir=os.listdir('./EuroSAT_RGB/EuroSAT_RGB/'), rootpath='./EuroSAT_RGB/EuroSAT_RGB/')
    dataloaders = createDataLoader(ds, 32)
    
    model = ResNet(num_classes=ds.num_classes, weights=ResNet18_Weights.DEFAULT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")