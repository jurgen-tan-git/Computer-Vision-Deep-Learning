import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks.boundary_attack import BoundaryAttack
import torch
from PIL import Image
import torchvision.transforms as transforms

torch.manual_seed(0)

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open("./mrshout2.jpg").convert('RGB')

    augmentation = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor()
    ])
    image = augmentation(image).unsqueeze(0).to(device)

    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    print(preds.item())

    foolbox_model = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=None)

    label = torch.tensor([preds.item()]).to(device)

    # apply the attack
    attack = BoundaryAttack(step_adaptation=0.5 , steps=250)
    epsilons = [
        0,
        1,
        10,
        50,
        100,
        500,
        1000
    ]
    raw_advs, clipped_advs, success = attack(foolbox_model, image, label, epsilons=epsilons)
    for i in range(len(epsilons)):
        print('-'*10)
        print("Episolon:", epsilons[i])
        print('-'*10)
        raw = raw_advs[i]
        clipped = clipped_advs[i]
        transforms.ToPILImage()(clipped.squeeze(0)).save("Task2_output_image_{}.png".format(epsilons[i]))

        output = model(clipped)
        _, preds = torch.max(output.data, 1)
        print("Predicted Class: ",preds.item())
        difference = clipped - raw

        difference = difference.squeeze(0)

        norm_inf = torch.linalg.norm(difference, ord=float('inf'), dim=(1,2))
        norm_1 = torch.linalg.norm(difference, ord=1, dim=(1,2))
        norm_2 = torch.linalg.norm(difference, ord=2, dim=(1,2))
        print("Infinity (inf) norm: {} {} {}".format(norm_inf[0], norm_inf[1], norm_inf[2]))
        print("Manhattan (L1) norm: {} {} {}".format(norm_1[0], norm_1[1], norm_1[2]))
        print("Euclidean (L2) norm: {} {} {}".format(norm_2[0], norm_2[1], norm_2[2]))




if __name__ == "__main__":
    main()
