import os
from torchvision import transforms
from pickle import dump
from tqdm.auto import tqdm
from numpy import float32
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import torch

torch.manual_seed(0)

class Utils:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.images = {}
        
    def split_data(self, test_size=0.2, val_size=0.2, random_state=0,multilabel=False):
            for folder in os.listdir(self.dir):
                for file in os.listdir(self.dir + folder):
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

            if multilabel == True:
                y_train = self.multilabel(y_train)
                y_val = self.multilabel(y_val)
                y_test = self.multilabel(y_test)

            return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    
    def multilabel(self, label):
        for i in range(len(label)):
            if label[i][0] == 1 or label[i][6] ==1:
                label[i][6] = 1
                label[i][0] = 1

            if label[i][1] == 1:
                label[i][2] = 1
        return label
        
        

        
    def getImages(self):
        return self.images
    
    def getTransorms(self):

        trans1=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trans2 = transforms.Compose([
        transforms.RandomRotation(degrees=10),  # Rotate by up to 10 degrees
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trans3 = color_contrast_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ColorJitter(hue=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform = [trans1, trans2, trans3]
        return self.transform
    
    def createDataLoaders(self, train_ds1, val_ds1, test_ds1, train_ds2=None, train_ds3=None, val_ds2=None, val_ds3=None, test_ds2=None, test_ds3=None, batchsize=32):

        if train_ds1 != None and train_ds2 != None and val_ds2 != None and val_ds3 != None:
            train1 = DataLoader(train_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
            train2 = DataLoader(train_ds2, batch_size=batchsize, shuffle=True, num_workers=4)
            train3 = DataLoader(train_ds3, batch_size=batchsize, shuffle=True, num_workers=4)

            val1 = DataLoader(val_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
            val2 = DataLoader(val_ds2, batch_size=batchsize, shuffle=True, num_workers=4)
            val3 = DataLoader(val_ds3, batch_size=batchsize, shuffle=True, num_workers=4)

            test1 = DataLoader(test_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
            test2 = DataLoader(test_ds2, batch_size=batchsize, shuffle=True, num_workers=4)
            test3 = DataLoader(test_ds3, batch_size=batchsize, shuffle=True, num_workers=4)

            train = [train1, train2, train3]
            val = [val1, val2, val3]
            test = [test1, test2, test3]

            return dict(train=train, val=val, test=test)
        else:
            train = DataLoader(train_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
            val = DataLoader(val_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
            test = DataLoader(test_ds1, batch_size=batchsize, shuffle=True, num_workers=4)
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
                # accuracy =  (  accuracy*datasize + self.check(preds, labels) ) / ( datasize + inputs.shape[0])
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        if labels[i][j] == 1:
                            class_pred = j
                    if class_pred not in self.avgprec:
                        self.avgprec[class_pred] = list()
                        self.avgprec[class_pred].append(average_precision_score(labels[i], cpuout[i]))
                    else:
                        self.avgprec[class_pred].append(average_precision_score(labels[i], cpuout[i]))


                    
                datasize += inputs.shape[0] #update datasize used in accuracy comp
        
        if criterion is None:   
            avgloss = None
        total = []
        print(self.avgprec.keys())
        with open('./Log/'+ self.name +'_ClassMAP.txt', 'a') as f:
            for i in range(len(self.avgprec)):
                try:
                    print("Class {} Mean Average Precision: {}".format(i, sum(self.avgprec[i])/len(self.avgprec[i])))
                    total.append(sum(self.avgprec[i])/len(self.avgprec[i]))
                    f.write("Class {} Mean Average Precision: {}\n".format(i, sum(self.avgprec[i])/len(self.avgprec[i])))
                except:
                    pass

            f.write("Total Mean Average Precision: {}\n\n".format(sum(total)/len(total)))
            f.close()
        measure = sum(total)/len(total)
        print("Total Mean Average Precision: {}".format(measure))
        return measure, avgloss

    def train_modelcv(self, dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device, lr, name, multilabel = False):

        self.name = name
        
        best_measure = 0
        best_epoch =-1
        train_loss_dict = {}
        val_loss_dict = {}


        for epoch in tqdm(range(num_epochs)):

            with open('./Log/'+ self.name +'_ClassMAP.txt', 'a') as f:
                f.write("Epoch: {}, Learning Rate: {}\n".format(epoch, lr))
                f.close()
                
            self.avgprec = {}
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            if multilabel == True:
                train_loss=self.train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
                measure, val_loss = self.evaluate(model, dataloader_cvtest, criterion = criterion, device = device)

                train_loss_dict[epoch] = train_loss
                val_loss_dict[epoch] = val_loss

                if measure > best_measure:
                    bestweights= model.state_dict()
                    best_measure = measure
                    best_epoch = epoch
                    transform_index = None
                    print('current best', measure, ' at epoch ', best_epoch)

            else:
                for i in range(3):
                    with open('./Log/'+ self.name +'_ClassMAP.txt', 'a') as f:
                        f.write("Augmentation: {}\n".format(i))
                        f.close()
                    train_loss=self.train_epoch(model,  dataloader_cvtrain[i],  criterion,  device , optimizer )
                    measure, val_loss = self.evaluate(model, dataloader_cvtrain[i], criterion = criterion, device = device)

                    train_loss_dict[str(epoch) + "A" + str(i)] = train_loss
                    val_loss_dict[str(epoch) + "A" + str(i)] = val_loss

                    if measure > best_measure:
                        bestweights= model.state_dict()
                        best_measure = measure
                        best_epoch = epoch
                        transform_index = i
                        print('current best', measure, ' at epoch ', best_epoch)

        print("Storing Losses")
        with open('./pkl/Task1_train_loss_' + str(lr) +  '.pkl', 'wb') as file:
            dump(train_loss_dict, file)

        with open('./pkl/Task1_val_loss_' + str(lr) +  '.pkl', 'wb') as file:
            dump(val_loss_dict, file)

        return best_epoch, best_measure, bestweights, transform_index
    
    def check(self, pred, label):
        count = 0
        for i in range(len(pred)):
            if label[i][pred[i]] == 1:
                count += 1
            else:
                pass
            
        return count