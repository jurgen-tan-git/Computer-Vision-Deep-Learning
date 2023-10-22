import os
import torch
from torchvision import transforms
from Task1 import CustomDataset
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset



if __name__ == "__main__":
    dirs = ['./valid-20231011T153034Z-001/valid/', './aquariumfishes/aquariumfishes/aquarium_pretrain/valid/']

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    for dir in dirs:
        print(dir)
        images = os.listdir(dir + 'images')
        labels = os.listdir(dir + 'labels')

        if dir.index == 0:
            dataset = CustomDataset(dir, images, labels, transform=transform)
        else:
            dataset = CustomDataset(dir, images, labels, transform=transform, fish=True)

        cocoAnnotation = get_coco_api_from_dataset(dataset)
        coco_evaluator2 = CocoEvaluator(cocoAnnotation, iou_types=['bbox'])
        print(len(dataset))

        for i in range(len(dataset)):
            image, target = dataset.__getitem__(i)
            predsdict = {
                "image_id": target["image_id"],
                "labels": target["labels"],
                "scores": torch.tensor([1.0]).repeat(target["labels"].shape[0]),
                "boxes": target["boxes"]
            }
            res = {target["image_id"]: predsdict}

            coco_evaluator2.update(res)
        coco_evaluator2.accumulate()
        coco_evaluator2.summarize()

