#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import FGSM, LinfPGD
import torch
from PIL import Image
import torchvision.transforms as transforms


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    device = torch.device("cuda:0")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    foolbox_model = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images = Image.open("./mrshout2.jpg").convert('RGB')

    augmentation = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor()
    ])

    images = augmentation(images).unsqueeze(0).to(device)
    print(images.shape)
    labels = torch.tensor([949]).to(device)

    clean_acc = accuracy(foolbox_model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = FGSM()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    raw_advs, clipped_advs, success = attack(foolbox_model, images, labels, epsilons=epsilons)
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - success.float().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # # we can also manually check this
    # # we will use the clipped advs instead of the raw advs, otherwise
    # # we would need to check if the perturbation sizes are actually
    # # within the specified epsilon bound
    # print()
    # print("we can also manually check this:")
    # print()
    # print("robust accuracy for perturbations with")
    # for eps, advs_ in zip(epsilons, clipped_advs):
    #     acc2 = accuracy(foolbox_model, advs_, labels)
    #     print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
    #     print("    perturbation sizes:")
    #     perturbation_sizes = (advs_.cpu() - images.cpu()).numpy().max(axis=(1, 2, 3))
    #     print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
    #     if acc2 == 0:
    #         break


if __name__ == "__main__":
    main()