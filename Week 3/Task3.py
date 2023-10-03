import os
import torchvision.transforms as transforms
from getimagenetclasses import get_classes, get_classlabel
from PIL import Image
from Task2 import evaluatemodel, check

def setting3(image, n):
    transform = transforms.Compose([
        transforms.Resize((n+28, n+28)),
        transforms.CenterCrop((n, n)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(image)

def main():

    dir = os.getcwd() + "/imagenet2500/imagespart"
    images = os.listdir(dir)[:1000]
    values = [ 128, 224, 288]
    
    for value in values:
        count=0
        correct=0
        for image in images:
            try:
                trans_image = Image.open(dir + "/" + image)

                trans_image = setting3(trans_image,value)

                trans_image = trans_image.unsqueeze(0)

                count+=1
                print(count)

                # Make predictions on the image
                outputs = evaluatemodel(trans_image)
                result = check(get_classlabel(image[:-5]), outputs[0])
                if result == True:
                    correct+=1


            except RuntimeError:
                print("Wrong Shape")
        print("Accuracy for " + str(value) + ": " + str(correct/count))
        open("Task3.txt", "a").write("Accuracy for " + str(value) + ": " + str(correct/count) + "\n")
        
    

if __name__ == "__main__":
    main()