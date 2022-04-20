import torch
from torch import nn

"""
Network structure:
(Conv 1)    Conv -> ReLU -> MaxPool ->
(Conv 2)    Conv -> ReLU -> MaxPool ->
(Linear 1)  Linear -> ReLU ->
(Linear 2)  Linear -> ReLU ->
(Linear 3)  Linear -> tanh

Parameters by layer of the network:
(Conv 1)
- frames_in:    Number of frames in input sequence
- H_in:         Height of input frames
- W_in:         Width of input frames
- kernels_1:    Number of kernels
- pad_1:        Amount of zero-padding
- kern_sz_1:    Size of each kernel
- pool_sz_1:    Size of max pooling region
(2)
- Same parameters as above, but suffixed with "_2"
(3)
- h3: Size hidden layer in first linear layer
(4)
- h4: Size of hidden layer in second linear layer
(5)
- frames_out:   Number of frames to output.
                Note: Size of layer = frames_out * H_in * W_in
"""

class RL_CNN(nn.Module):
    def __init__(self, frames_in=10, H_in=64, W_in=64,
                 kernels_1=30, pad_1=1, kern_sz_1=5, stride_1=1,
                 pool_sz_1=2,
                 kernels_2=60, pad_2=1, kern_sz_2=5, stride_2=1,
                 pool_sz_2=2,
                 h_3=4096, h_4=4096, frames_out=1):
        super().__init__()

        # Conv 1
        self.conv1 = nn.Conv2d(frames_in, kernels_1, kern_sz_1, padding=pad_1)
        self.pool1 = nn.MaxPool2d(pool_sz_1, pool_sz_1)

        H_1_pool = (H_in + 2*pad_1 - kern_sz_1 + 1) // (stride_1 * pool_sz_1)
        W_1_pool = (W_in + 2*pad_1 - kern_sz_1 + 1) // (stride_1 * pool_sz_1)

        # Conv 2
        self.conv2 = nn.Conv2d(kernels_1, kernels_2, kern_sz_2, padding=pad_2)
        self.pool2 = nn.MaxPool2d(pool_sz_2, pool_sz_2)

        H_2_pool = (H_1_pool + 2*pad_2 - kern_sz_2 + 1) // (stride_2 * pool_sz_2)
        W_2_pool = (W_1_pool + 2*pad_2 - kern_sz_2 + 1) // (stride_2 * pool_sz_2)

        # Linear Layers
        self.fc1 = nn.Linear(kernels_2 * H_2_pool * W_2_pool, h_3)
        self.fc2 = nn.Linear(h_3, h_4)
        self.fc3 = nn.Linear(h_4, frames_out * H_in * W_in)

    def __call__(self, x):

        # Conv 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Conv 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Linear Layers
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

def init_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_normal_(m.weight.data)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)