import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.optim as optim
from Utils import Utils
from Task1 import CustomImageDataset
from pickle import dump
from torchvision import transforms

torch.manual_seed(0)

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

    dir = './EuroSAT_RGB/'
    util = Utils(dir)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = util.split_data(multilabel=True)
    print(num_classes)
    image_dict = util.getImages()
    tranform = util.getTransorms()

    test_transform = util.getTestTransforms()
            
    
    train_ds = CustomImageDataset(dir,X_train, y_train, image_dict, transform=tranform[0])
    
    val_ds = CustomImageDataset(dir, X_val, y_val, image_dict, transform=tranform[0])

    test_ds = CustomImageDataset(dir, X_test, y_test, image_dict,  transform=test_transform)

    dataloaders = util.createDataLoaders(train_ds, val_ds, test_ds)

    device = torch.device("cuda:0")
    
    with open('./Log/multilabel-model_ClassMAP.txt', 'w') as f:
        f.write("Class Mean Average Precision\n")
        f.close()
    epochs = 10
    learning_rates = [0.1, 0.01]
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    loss = torch.nn.BCEWithLogitsLoss()
    
    for lr in learning_rates:
        model = MultiLabelModel(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT).to(device)
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9) # which parameters to optimize during training?
        _, best_perfmeasure, bestweights, _  = util.train_modelcv(dataloader_cvtrain = dataloaders['train'],
                                                                dataloader_cvtest = dataloaders['val'] ,
                                                                model = model,
                                                                criterion = loss, 
                                                                optimizer = optimizer, 
                                                                scheduler = None, 
                                                                num_epochs = epochs, 
                                                                device = device,
                                                                lr = lr,
                                                                name='multilabel-model',
                                                                multilabel=True)
        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    model.load_state_dict(weights_chosen)
    
    with open('./Model/Task2_Model'  +'.pkl', 'wb') as file:
        dump(bestweights, file)
    
    accuracy,_ = util.evaluate(model = model , dataloader= dataloaders['test'], criterion = None, device = device)
    print('best hyperparameter', best_hyperparameter)
    print('best measure', bestmeasure)
    print('accuracy val',bestmeasure)
    print('accuracy test',accuracy) 
    with open('./Log/multilabel-model_ClassMAP.txt', 'a') as f:
        f.write("Best Hyperparameter: " + str(best_hyperparameter) + "\n")
        f.close()
    
        