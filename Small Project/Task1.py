import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.optim as optim
from pickle import dump
from Utils import Utils

torch.manual_seed(0)

class CustomImageDataset(Dataset):
    def __init__(self, dir, files,  labels, image_dict, transform=None, random_state=0):
        self.dir = dir
        self.files = files
        self.image_dict = image_dict
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir, self.image_dict[self.files[idx]])
        img_name = os.path.join(img_name, self.files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class Model(torch.nn.Module):
    def __init__(self, num_classes, weights=None):
        super(Model, self).__init__()
        self.model = resnet50(weights=weights)
        
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


    
if __name__ == '__main__':
    dir = './EuroSAT_RGB/EuroSAT_RGB/'
    util = Utils(dir)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = util.split_data()
    image_dict = util.getImages()
    tranform = util.getTransorms()
    
    train_ds1 = CustomImageDataset(dir,X_train, y_train, image_dict, transform=tranform[0])
    train_ds2= CustomImageDataset(dir,X_train, y_train, image_dict, transform=tranform[1])
    train_ds3 = CustomImageDataset(dir,X_train, y_train, image_dict, transform=tranform[2])
    
    val_ds1 = CustomImageDataset(dir, X_val, y_val, image_dict, transform=tranform[0])
    val_ds2 = CustomImageDataset(dir, X_val, y_val, image_dict, transform=tranform[1])
    val_ds3 = CustomImageDataset(dir, X_val, y_val, image_dict, transform=tranform[2])

    test_ds1 = CustomImageDataset(dir, X_test, y_test, image_dict,  transform=tranform[0])
    test_ds2 = CustomImageDataset(dir, X_test, y_test, image_dict,  transform=tranform[1])
    test_ds3 = CustomImageDataset(dir, X_test, y_test, image_dict,  transform=tranform[2])

    dataloaders = util.createDataLoaders(train_ds1=train_ds1, train_ds2=train_ds2, train_ds3=train_ds3, 
                                         val_ds1=val_ds1, val_ds2=val_ds2, val_ds3=val_ds3, 
                                         test_ds1=test_ds1, test_ds2=test_ds2, test_ds3=test_ds3)
    
    device = torch.device("cuda:0")
    
    
    with open('./Log/Task1_ClassMAP.txt', 'w') as f:
        f.write("Class Mean Average Precision\n")
        f.close()
    epochs = 15
    learning_rates = [0.1, 0.01]
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    loss = torch.nn.CrossEntropyLoss()
    
    for lr in learning_rates:
        model = Model(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT).to(device)
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9) # which parameters to optimize during training?
        _, best_perfmeasure, bestweights, transform_index = util.train_modelcv(dataloader_cvtrain = dataloaders['train'],
                                                                dataloader_cvtest = dataloaders['val'] ,
                                                                model = model,
                                                                criterion = loss, 
                                                                optimizer = optimizer, 
                                                                scheduler = None, 
                                                                num_epochs = epochs, 
                                                                device = device,
                                                                lr = lr,
                                                                name='mutliclass-model')
        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    model.load_state_dict(weights_chosen)
    
    with open('./Model/Task1_Model'  +'.pkl', 'wb') as file:
            dump(bestweights, file)


    accuracy,_ = util.evaluate(model = model , dataloader= dataloaders['test'][transform_index], criterion = None, device = device)
    print('best hyperparameter', best_hyperparameter)
    print("Best Augmentation Index", transform_index)
    print('accuracy val',bestmeasure)
    print('accuracy test',accuracy) 
    
    