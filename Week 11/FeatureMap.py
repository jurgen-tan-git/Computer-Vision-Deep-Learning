import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Load a pretrained ResNet-18 model
pretrained_model = models.resnet18(pretrained=True)
pretrained_model.eval()

# Define the forward hook function
feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        # Process the feature map
        processed_feature_map = process_feature_map(output)
        feature_maps[name] = processed_feature_map
    return hook

def process_feature_map(feature_map):
    # Set negative activations to zero
    feature_map[feature_map < 0] = 0
    
    # Take the standard deviation over spatial dimensions
    std_feature_map = torch.std(feature_map, dim=(2, 3))
    
    return std_feature_map

# Register forward hooks to the desired layers
target_layers = {
    'firstconv': pretrained_model.conv1,
    'lastconv': pretrained_model.layer4[-1].conv2
}

hooks = []
for name, layer in target_layers.items():
    hook = layer.register_forward_hook(hook_fn(name))
    hooks.append(hook)

# Load and preprocess a batch of images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_folderpath = "./imagenet2500/imagespart"
image_paths = [os.path.join(image_folderpath, path) for path in os.listdir(image_folderpath)]
images = [transform(Image.open(path).convert("RGB")) for path in image_paths]
images = torch.stack(images)

# Forward pass through the model
with torch.no_grad():
    outputs = pretrained_model(images)

# Unregister the forward hooks
for hook in hooks:
    hook.remove()

# Extracted and processed feature maps are now stored in the 'feature_maps' dictionary
# The keys are the names of the target layers, and the values are the processed feature maps

# Save the processed feature maps to a file
save_path = "./processed_feature_maps.pth"
torch.save(feature_maps, save_path)
print(f"Processed feature maps saved to {save_path}")

# Example: Print the shape of the processed feature maps
for name, feature_map in feature_maps.items():
    print(f"Processed Feature map {name} shape: {feature_map.shape}")
