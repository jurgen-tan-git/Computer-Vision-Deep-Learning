import torch
import torchvision.models as models

# Load a pretrained ResNet-18 model
pretrained_model = models.resnet18(pretrained=True)
pretrained_model.eval()

# Access the first convolutional layer (conv1)
first_conv_layer = pretrained_model.conv1
print("First Convolutional Layer:", first_conv_layer)

# Access the last convolutional layer in the final block (layer4)
last_conv_layer = pretrained_model.layer4[-1].conv2
print("Last Convolutional Layer:", last_conv_layer)
