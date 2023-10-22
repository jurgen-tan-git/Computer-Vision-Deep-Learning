import os 
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(0)

class CustomDataset(Dataset):

    def __init__(self, dir, images, labels, transform=None, fish=False):
        self.dir = dir
        self.image_names = images
        self.label_names = labels
        self.transform = transform
        self.fish = fish

    def __getitem__(self, idx):
        path = os.path.join(self.dir, 'images', self.image_names[idx])
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        if self.fish:
            image = transforms.Resize(256)(image)
        with open(self.dir + '/labels/' + self.label_names[0][:-3] + 'txt') as f:
            labels = f.readlines()

        label_list = []
        box_list = []
        area_list = []

        for i in range(len(labels)):
            label_txt = labels[i].split(' ')

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
    
if __name__ == '__main__':
    dir = './aquariumfishes/aquariumfishes/aquarium_pretrain/train/'
    images = os.listdir(dir + 'images')
    labels = os.listdir(dir + 'labels')

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = CustomDataset(dir, images, labels, transform=transform, fish=True)

    items = [2,9]
    for item in items:
        image, target = dataset.__getitem__(item)
        print(image.shape)
        print(target['labels'])
        print(target['boxes'])
