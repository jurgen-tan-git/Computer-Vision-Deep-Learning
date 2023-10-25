import os
import torch
from torchvision import transforms
from Task1 import CustomDataset, getTransform
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset



if __name__ == "__main__":
    dirs = ['./valid-20231011T153034Z-001/valid/', './aquariumfishes/aquariumfishes/aquarium_pretrain/valid/']

    transform = getTransform()
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

            centered_box = []
            for value in target["boxes"]:
                x_start, y_start, x_end, y_end = value

                x_center = (x_start + x_end) / 2
                y_center = (y_start + y_end) / 2


                centered_box.append(torch.tensor([x_center-0.1, y_center-0.1, x_center+0.1, y_center+0.1]))

                # width = x_end - x_start
                # height = y_end - y_start

                # centered_box.append(torch.tensor([0.5-width/2, 0.5, 0.5+width/2, 0.5]))

            predsdict = {
                "image_id": target["image_id"],
                "labels": target["labels"],
                "scores": torch.tensor([1.0]).repeat(target["labels"].shape[0]),
                "boxes": torch.cat(centered_box).reshape(-1, 4)
            }
            res = {target["image_id"]: predsdict}

            coco_evaluator2.update(res)
        coco_evaluator2.accumulate()
        coco_evaluator2.summarize()

