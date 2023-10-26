import os
import torch
from PIL import Image
from ultralytics import YOLO
from Task1 import getTransform
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):

    def __init__(self, dir, images, labels, transform=None, fish=False):
        self.dir = dir
        self.image_names = images
        self.label_names = labels
        self.transform = transform
        self.fish = fish
        self.index = []

        for i in range(len(self.image_names)):
            with open(self.dir + '/labels/' + self.label_names[i][:-3] + 'txt') as f:
                labels = f.readlines()
            for j in range(len(labels)):
                if int(labels[j].split(' ')[0]) in [15, 7, 14, 17, 20] and i not in self.index:
                    self.index.append(i)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, 'images', self.image_names[idx])
        image = Image.open(path)

        if self.transform:
            image = self.transform(image).to(torch.float32)

        if self.fish:
            image = transforms.Resize(256)(image)

        with open(self.dir + '/labels/' + self.label_names[idx][:-3] + 'txt') as f:
            labels = f.readlines()

        label_list = []
        box_list = []
        area_list = []

        for i in range(len(labels)):
            label_txt = labels[i].split(' ')

             # Stop Sign Mapping: index 15 from traffic dataset to index 11 in coco dataset
            if label_txt[0] == '15':
                label_txt[0] = 11

            # Traffic Light Mapping: index 7, 14, 17, 20 to index 9 in coco dataset
            elif label_txt[0] in ['7', '14', '17', '20']:
                label_txt[0] = 9

            else:
                label_txt[0] = -1

            label_list.append(int(label_txt[0]))

            x_center = float(label_txt[1])
            y_center = float(label_txt[2])
            x_width = float(label_txt[3])
            y_height = float(label_txt[4])

            # Calculate the coordinates of the upper left corner and the lower right corner
            x_start = x_center - x_width/2
            y_start = y_center - y_height/2
            x_end = x_center + x_width/2
            y_end = y_center + y_height/2

            box = torch.tensor([x_start, y_start, x_end, y_end])
            box_list.append(box)

            # Calculate the area
            area = x_width * y_height
            area_list.append(area)

        targetdic = dict()
        targetdic['image_id'] = self.image_names[idx][:-4]
        targetdic['iscrowd'] = torch.zeros((len(labels)), dtype=torch.bool)
        targetdic['boxes'] = torch.cat(box_list).reshape(-1, 4)
        targetdic['labels'] = torch.tensor(label_list)
        targetdic['area'] = torch.tensor(area_list)
        
        return image, targetdic
    
    def __len__(self):
        return len(self.image_names)
    
    def getindex(self):
        return self.index
    

if __name__ == "__main__":
    dir = './valid-20231011T153034Z-001/valid/'
    images = os.listdir(dir + 'images')
    labels = os.listdir(dir + 'labels')

    transform = getTransform()
    dataset = CustomDataset(dir, images, labels, transform=transform)

    cocoAnnotation = get_coco_api_from_dataset(dataset)
    coco_evaluator = CocoEvaluator(cocoAnnotation, iou_types=['bbox'])

    device = torch.device("cuda")
    model = YOLO("yolov8x.pt").to(device)
    
    index = dataset.getindex()

    res = {}
    for value in index:
        image, target = dataset.__getitem__(value)
        image = image.unsqueeze(0).to(device)

        result = model.predict(image)
        result = result[0]

        if len(result.boxes) > 0:
            labels = torch.tensor(result.boxes.cls)
            scores = torch.tensor(result.boxes.conf)
            boxes = torch.tensor(result.boxes.xyxyn)
            

            predsdict = {
                    "image_id": target["image_id"],
                    "labels": labels,
                    "scores": scores,
                    "boxes": boxes
                }

            
            res[target["image_id"]] = predsdict
    
    coco_evaluator.update(res)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()





        