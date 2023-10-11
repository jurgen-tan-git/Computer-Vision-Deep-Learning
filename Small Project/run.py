import os
from PIL import Image
from numpy import float32
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
from pickle import dump
from tqdm.auto import tqdm

class Utils:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.images = {}
        
    def split_data(self, test_size=0.2, val_size=0.2, random_state=0):
            for folder in os.listdir(dir):
                for file in os.listdir(dir + folder):
                    self.images[file] = folder
                    
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                list(self.images.keys()), list(self.images.values()), test_size=test_size, random_state=random_state
            )
            # Split the train data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state
            )
            # Binarize labels using LabelBinarizer
            label_binarizer = LabelBinarizer()
            y_train = label_binarizer.fit_transform(y_train).astype(float32)
            y_val = label_binarizer.transform(y_val).astype(float32)
            y_test = label_binarizer.transform(y_test).astype(float32)
            num_classes = len(label_binarizer.classes_)
            print(label_binarizer.classes_)
            return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
        
    def getImages(self):
        return self.images
    
    def createDataLoaders(self, train_ds, val_ds, test_ds):
        train = DataLoader(train_ds, batch_size=3, shuffle=True)
        val = DataLoader(val_ds, batch_size=3, shuffle=True)
        test = DataLoader(test_ds, batch_size=3, shuffle=True)
        return dict(train=train, val=val, test=test)
    
    def train_epoch(self, model, trainloader, criterion, device, optimizer):
        model.train()  

        for batch_idx, data in enumerate(trainloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()  
        return loss.item() 

    def evaluate(self, model, dataloader, criterion, device):

        model.eval() 


        with torch.no_grad():
        
            datasize = 0
            accuracy = 0
            avgloss = 0
            for ctr, data in enumerate(dataloader):

                
                inputs = data[0].to(device)        
                outputs = model(inputs)

                labels = data[1]

                # computing some loss
                cpuout= outputs.to('cpu')
                if criterion is not None:
                    curloss = criterion(cpuout, labels)
                    avgloss = ( avgloss*datasize + curloss ) / ( datasize + inputs.shape[0])

                # for computing the accuracy
                labels = labels.float()
                _, preds = torch.max(cpuout, 1) # get predicted class 
                accuracy =  (  accuracy*datasize + self.check(preds, labels) ) / ( datasize + inputs.shape[0])
                    
                datasize += inputs.shape[0] #update datasize used in accuracy comp
        
        if criterion is None:   
            avgloss = None
            
        return accuracy, avgloss

    def train_modelcv(self, dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device, lr):

        best_measure = 0
        best_epoch =-1
        train_loss_dict = {}
        val_loss_dict = {}
        val_acc_dict = {}

        for epoch in tqdm(range(num_epochs)):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            train_loss=self.train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
            train_loss_dict[epoch] = train_loss

            measure, val_loss = self.evaluate(model, dataloader_cvtest, criterion = criterion, device = device)
            val_loss_dict[epoch] = val_loss
            val_acc_dict[epoch] = measure
            

            print('perfmeasure', measure )

            # store current parameters because they are the best or not?
            if measure > best_measure: # > or < depends on higher is better or lower is better?
                bestweights= model.state_dict()
                best_measure = measure
                best_epoch = epoch
                print('current best', measure, ' at epoch ', best_epoch)

        return best_epoch, best_measure, bestweights
    
    def check(self, pred, label):
        count = 0
        for i in range(len(pred)):
            if pred[i] == label[i][pred[i]]:
                count += 1
            else:
                pass
            
        return count

class CustomImageDataset(Dataset):
    def __init__(self, dir, files, labels, transform=None, random_state=0):
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


class ResNet(torch.nn.Module):
    def __init__(self, num_classes, weights=None):
        super(ResNet, self).__init__()
        self.model = resnet18(weights=weights)  # Load the pre-trained ResNet-50 model

        # Modify the final fully connected layer (fc) for your custom classification task
        num_features = self.model.fc.in_features  # Get the number of input features to the final layer
        self.model.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


    
if __name__ == '__main__':
    dir = './EuroSAT_RGB/EuroSAT_RGB/'
    util = Utils(dir)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = util.split_data()
    image_dict = util.getImages()
    tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    train_ds = CustomImageDataset(dir,X_train, y_train, transform=tranform)
    val_ds = CustomImageDataset(dir, X_val, y_val, transform=tranform)
    test_ds = CustomImageDataset(dir, X_test, y_test, transform=tranform)

    dataloaders = util.createDataLoaders(train_ds, val_ds, test_ds)
    
    device = torch.device("cuda:0")
    model = ResNet(num_classes=num_classes, weights=ResNet18_Weights.DEFAULT).to(device)
    
    epochs = 10
    learning_rates = [0.1, 0.01, 0.001]
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
                                                                lr = lr)
        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    model.load_state_dict(weights_chosen)

    accuracy,_ = util.evaluate(model = model , dataloader= dataloaders['test'], criterion = None, device = device)
    print('accuracy val',bestmeasure.item())
    print('accuracy test',accuracy.item()) 
    
    