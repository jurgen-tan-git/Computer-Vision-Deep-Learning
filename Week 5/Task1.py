import os
from PIL import Image

from time import process_time
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
from pickle import dump
from tqdm.auto import tqdm
from PlotLoss import plot


class CustomImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.transform = transform

        with open(txt_file, 'r') as file:
            self.image_names = file.read().splitlines()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx].split(" ")[0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, int(self.image_names[idx].split(" ")[1])

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


def train_epoch(model, trainloader, criterion, device, optimizer):
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


def evaluate(model, dataloader, criterion, device):

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
          accuracy =  (  accuracy*datasize + torch.sum(preds == labels) ) / ( datasize + inputs.shape[0])
            
          datasize += inputs.shape[0] #update datasize used in accuracy comp
    
    if criterion is None:   
      avgloss = None
          
    return accuracy, avgloss


def train_modelcv(dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device, lr, name):

    best_measure = 0
    best_epoch =-1
    train_loss_dict = {}
    val_loss_dict = {}
    val_acc_dict = {}

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
        train_loss_dict[epoch] = train_loss

        measure, val_loss = evaluate(model, dataloader_cvtest, criterion = criterion, device = device)
        val_loss_dict[epoch] = val_loss
        val_acc_dict[epoch] = measure
        

        print('perfmeasure', measure.item() )

        # store current parameters because they are the best or not?
        if measure > best_measure: # > or < depends on higher is better or lower is better?
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure.item(), ' at epoch ', best_epoch)

    with open('./pkl/' + name + '_train_loss_' + str(lr) +  '.pkl', 'wb') as file:
            dump(train_loss_dict, file)

    with open('./pkl/' + name + '_val_loss_' + str(lr) +  '.pkl', 'wb') as file:
            dump(val_loss_dict, file)

    with open('./pkl/' + name + '_val_acc_' + str(lr) +  '.pkl', 'wb') as file:
            dump(val_acc_dict, file)

    plot(train_loss_dict, val_loss_dict, val_acc_dict,name + '_plot_' + str(lr) + '.png')
    return best_epoch, best_measure, bestweights


def getTransforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_data_loaders(batch_size:int, transforms:dict):
    dir = './102flowers/flowers_data/jpg'

    train_dataloader = DataLoader(CustomImageDataset(txt_file= dir[:-3] + 'trainfile.txt',
                                                    root_dir=dir,
                                                    transform=transforms['train']),
                                                batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(CustomImageDataset(txt_file=dir[:-3] + 'valfile.txt',
                                                    root_dir=dir,
                                                    transform=transforms['val']),
                                                batch_size=batch_size, shuffle=False)
    
    test_dataloader = DataLoader(CustomImageDataset(txt_file=dir[:-3] + 'f',
                                                    root_dir=dir,
                                                    transform=transforms['train']),
                                                batch_size=batch_size, shuffle=False)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

def last2layerparams(model):
    params_to_optimize = []
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name:
            params_to_optimize.append(param)
    return params_to_optimize

def getlayerparams(model):
    params_to_optimize = []
    for _, param in model.named_parameters():
        params_to_optimize.append(param)
    return params_to_optimize

def run(model, optim_layers, loss, name):
    maxnumepochs=15 
    lrates=[0.01, 0.001]
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    

    for lr in lrates:
        optimizer = optim.SGD(params=optim_layers, lr=lr, momentum=0.9) # which parameters to optimize during training?
        _, best_perfmeasure, bestweights,  = train_modelcv(dataloader_cvtrain = dataloader['train'],
                                                                dataloader_cvtest = dataloader['val'] ,
                                                                model = model,
                                                                criterion = loss, 
                                                                optimizer = optimizer, 
                                                                scheduler = None, 
                                                                num_epochs = maxnumepochs, 
                                                                device = device,
                                                                lr = lr,
                                                                name=name)
        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    model.load_state_dict(weights_chosen)

    accuracy,_ = evaluate(model = model , dataloader= dataloader['test'], criterion = None, device = device)
    print('accuracy val',bestmeasure.item())
    print('accuracy test',accuracy.item()) 

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = create_data_loaders(batch_size=32, transforms=getTransforms())
    loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    models = [ResNet(weights=None).to(device),
              ResNet(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)]

    # print("Running Setting A")
    run(model= models[0], optim_layers=getlayerparams(models[0]), loss=loss, name="Setting_A")

    print("Running Setting B")
    run(model= models[1], optim_layers=getlayerparams(models[1]), loss=loss, name="Setting_B")

    print("Running Setting C2")
    run(model=models[1], optim_layers=last2layerparams(models[1]), loss=loss, name="Setting_C2")