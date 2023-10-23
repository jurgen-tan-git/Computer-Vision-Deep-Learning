import os
import torch
from PIL import Image
from ultralytics import YOLO
from Task1 import CustomDataset, getTransform


if __name__ == "__main__":
    dir = './valid-20231011T153034Z-001/valid/'
    images = os.listdir(dir + 'images')
    labels = os.listdir(dir + 'labels')

    transform = getTransform()

    dataset = CustomDataset(dir, images, labels)

    device = torch.device("cuda")
    model = YOLO("yolov8s.pt").to(device)
    
    images=[]
    for i in range(dataset.__len__()):
        image, target = dataset.__getitem__(i)
        images.append(image)
    results = model.predict(images, save=True, device=device, save_txt=True, save_crop=True)



        