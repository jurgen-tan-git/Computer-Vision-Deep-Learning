import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from getimagenetclasses import get_classes, get_classlabel
from Task2 import check
from PIL import Image


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

    return probabilities

def label_to_class(probabilities):
    labels = get_classes()
    prob, prob_class = torch.topk(probabilities, 1)
    class_index = prob_class[0].item()
    class_label = labels[class_index]
    probability = prob[0].item() * 100
    return class_label, probability

def setting4(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    transformed_crops = []
    for crop in transforms.FiveCrop((224, 224))(image):
        transformed_crop = transform(crop)
        transformed_crops.append(transformed_crop)

    # Return a list of transformed crops
    return transformed_crops


def main():

    dir = os.getcwd() + "/imagenet2500/imagespart"
    images = os.listdir(dir)[:1000]
    correct=0
    count=0
    for image in images:
        try:
            trans_image = Image.open(dir + "/" + image)

            trans_image = setting4(trans_image)
            predictions = []

            count+=1
            print(count)

            for crop in trans_image:
                crop = crop.unsqueeze(0)
                outputs = evaluatemodel(crop)
                predictions.append(outputs)
            avg_prob = sum(predictions) / len(predictions)
            avg_class, avg_prob = label_to_class(avg_prob)
            result = check(get_classlabel(image[:-5]), avg_class)
            if result == True:
                correct+=1

        except:
            print("Wrong Shape")
            pass
    open("Task4.txt", "w").write("Five Crop Accuracy: " + str(correct/count))
    

if __name__ == "__main__":
    main()