import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.optim as optim
from Utils import Utils
from Task1 import CustomImageDataset


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

    image_dict = util.getImages()
    tranform = util.getTransorms()
    
    train_ds = CustomImageDataset(dir,X_train, y_train, image_dict, transform=tranform[0])
    
    val_ds = CustomImageDataset(dir, X_val, y_val, image_dict, transform=tranform[0])

    test_ds = CustomImageDataset(dir, X_test, y_test, image_dict,  transform=tranform[0])

    dataloaders = util.createDataLoaders(train_ds, val_ds, test_ds)

    
    with open('./ClassMAP.txt', 'w') as f:
        f.write("Class Mean Average Precision\n")
        f.close()
    epochs = 15
    learning_rates = [0.1,0.01, 0.001]
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    for lr in learning_rates:
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9) # which parameters to optimize during training?
        _, best_perfmeasure, bestweights,  = util.train_modelcv(dataloader_cvtrain = dataloaders['train'],
                                                                dataloader_cvtest = dataloaders['val'] ,
                                                                model = model,
                                                                criterion = loss, 
                                                                optimizer = optimizer, 
                                                                scheduler = None, 
                                                                num_epochs = epochs, 
                                                                device = device,
                                                                lr = lr,
                                                                name='resnet50',
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
    print('best hyperparameter', best_hyperparameter)
    print('best measure', bestmeasure)


    accuracy,_ = util.evaluate(model = model , dataloader= dataloaders['test'], criterion = None, device = device)
    print('accuracy val',bestmeasure)
    print('accuracy test',accuracy) 
    
    
        