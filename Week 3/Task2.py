import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from getimagenetclasses import get_classes, get_classlabel




def evaluatemodel(image):

    # Load the pretrained ResNet-18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to('cpu')

    # Set the model to evaluation mode
    model.eval()

    # Make predictions on the image
    with torch.no_grad():
        outputs = model(image)


    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    labels = get_classes()

    prob, output_class = torch.topk(probabilities, 1)
    class_index = output_class[0].item()
    class_label = labels[class_index]
    probability = prob[0].item() * 100
    return class_label, probability

def standard_setting(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(image)

def standard_setting(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(image)

def setting1(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0, 0, 0],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(image)

def setting2(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[1, 1, 1]
        )
    ])
    return transform(image)

def check(actual, predicted):
    if actual == predicted:
        return True
    else:
        return False


def main():

    count=0
    standard_correct = 0
    setting1_correct = 0
    setting2_correct = 0
    dir = os.getcwd() + "/imagenet2500/imagespart"
    images = os.listdir(dir)[:1000]
    
    for image in images:
        open_image = Image.open(dir + "/" + image)
        try:
            func = [standard_setting(open_image), setting1(open_image), setting2(open_image)]
            count+=1
            print(count)

            for i in range(len(func)):       
                trans_image = func[i]
                trans_image = trans_image.unsqueeze(0)

                # Make predictions on the image
                outputs = evaluatemodel(trans_image)
                result = check(get_classlabel(image[:-5]), outputs[0])

                if result == True:
                    if i == 0:
                        standard_correct += 1
                    elif i == 1:
                        setting1_correct += 1
                    elif i == 2:
                        setting2_correct += 1
                    else:
                        pass
        except:
            pass
    print("Standard Setting Accuracy: " + str(standard_correct/count) + '\n' + 
                                 "Setting 1 Accuracy: " + str(setting1_correct/count) + '\n' +
                                 "Setting 2 Accuracy: " + str(setting2_correct/count))
    
    open("Task2.txt", "w").write("Standard Setting Accuracy: " + str(standard_correct/count) + '\n' + 
                                 "Setting 1 Accuracy: " + str(setting1_correct/count) + '\n' +
                                 "Setting 2 Accuracy: " + str(setting2_correct/count))

if __name__ == '__main__':
    main()  