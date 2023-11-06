import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

def loadimage2tensor(nm, resize=300, mean= [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]):
    image = Image.open(nm).convert('RGB')
    image = transforms.Resize(resize)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean, std)(image)
    return image

def invert_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Create a copy of the tensor to avoid modifying the original
    inverted_tensor = tensor.clone()

    for t, m, s in zip(inverted_tensor[0], mean, std):
        t.mul_(s).add_(m)

    return inverted_tensor



device = torch.device("cuda:0")
imorig = loadimage2tensor('mrshout2.jpg').unsqueeze(0).to(device)

targetclass = 949
stepsize = 1
count = 0

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
with torch.no_grad():
    outputs = model(imorig)
    _, preds = torch.max(outputs.data, 1)
print(outputs.data[0,preds[0].item()],preds.item())
tobechanged = imorig.clone()
tobechanged.requires_grad = True

currentprediction = preds.item()

while currentprediction!=targetclass:
    count += 1

    outputs = model(tobechanged)

    score= outputs[0,targetclass] #objective to be optimized
    score.backward()

    tmptensor= tobechanged.data + stepsize * tobechanged.grad #update in temporary variable
    unscaledimage = invert_normalize(tmptensor) #determine where it would get out of bounds

    tobechanged.grad[(unscaledimage<1./255.0)|(unscaledimage > 254./255.0)]=0 #set grad to zero where it would get out of bounds
    tobechanged.data+=stepsize * tobechanged.grad #apply gradient descent
    
    tensor_image = invert_normalize(tobechanged.data)
    save_image(tensor_image[0], 'output_image.jpg')

    tobechanged.grad.zero_() # erase used gradient

    outputs = model(tobechanged) # update prediction




    with torch.no_grad():
        _, preds = torch.max(outputs.data, 1)
        print('in iter',preds.item(), outputs.data[0,preds[0].item()].item(),outputs.data[0,targetclass].item())
    currentprediction = preds[0].item()

print("Iterations taken", count)

image = loadimage2tensor('output_image.jpg').unsqueeze(0).to(device)
outputs = model(image)
_, preds = torch.max(outputs.data, 1)
print(outputs.data[0,preds[0].item()],preds.item())