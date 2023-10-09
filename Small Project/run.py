import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


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
    

if __name__ == '__main__':
    ds = CustomImageDataset(dir=os.listdir('./EuroSAT_RGB/EuroSAT_RGB/'), rootpath='./EuroSAT_RGB/EuroSAT_RGB/')
