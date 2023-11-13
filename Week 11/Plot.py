import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Load the saved feature maps
feature_maps_path = "./processed_feature_maps.pth"
feature_maps = torch.load(feature_maps_path)

# Analyze the firstconv feature map
firstconv_feature_map = feature_maps['firstconv']

std_per_channel_firstconv = torch.std(firstconv_feature_map, dim=(0,))
sorted_indices_firstconv = torch.argsort(std_per_channel_firstconv, descending=True)
sorted_std_firstconv = std_per_channel_firstconv[sorted_indices_firstconv]
normalized_std_firstconv = sorted_std_firstconv / torch.sum(sorted_std_firstconv)
cumulative_sum_firstconv = torch.cumsum(normalized_std_firstconv, dim=0)

# Analyze the lastconv feature map
lastconv_feature_map = feature_maps['lastconv']

std_per_channel_lastconv = torch.std(lastconv_feature_map, dim=(0,))
sorted_indices_lastconv = torch.argsort(std_per_channel_lastconv, descending=True)
sorted_std_lastconv = std_per_channel_lastconv[sorted_indices_lastconv]
normalized_std_lastconv = sorted_std_lastconv / torch.sum(sorted_std_lastconv)
cumulative_sum_lastconv = torch.cumsum(normalized_std_lastconv, dim=0)

# Plot both side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot firstconv
axs[0].plot(cumulative_sum_firstconv.numpy(), color='orange')
axs[0].set_title('Cumulative Sum - first convolution layer')
axs[0].set_xlabel('Channel Index')
axs[0].set_ylabel('Cumulative Sum')

# Plot lastconv
axs[1].plot(cumulative_sum_lastconv.numpy(), color='blue')
axs[1].set_title('Cumulative Sum - last convolution layer')
axs[1].set_xlabel('Channel Index')
axs[1].set_ylabel('Cumulative Sum')

plt.show()
