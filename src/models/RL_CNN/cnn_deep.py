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


class RL_CNN_DEEP(nn.Module):
    def __init__(self, frames_in=10, H_in=64, W_in=64,
                 kernels_1=15, pad_1=2, kern_sz_1=5, stride_1=1,
                 pool_sz_1=2,
                 kernels_2=30, pad_2=2, kern_sz_2=5, stride_2=1,
                 pool_sz_2=2,
                 kernels_3=60, pad_3=1, kern_sz_3=3, stride_3=1,
                 pool_sz_3=1,
                 kernels_4=60, pad_4=1, kern_sz_4=3, stride_4=1,
                 pool_sz_4=1,
                 h_5=6144, h_6=4096, frames_out=1):
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

        # Conv 3
        self.conv3 = nn.Conv2d(kernels_2, kernels_3, kern_sz_3, padding=pad_3)
        # self.pool3 = nn.MaxPool2d(pool_sz_3, pool_sz_3)

        H_3_pool = (H_2_pool + 2*pad_3 - kern_sz_3 + 1) // (stride_3 * pool_sz_3)
        W_3_pool = (W_2_pool + 2*pad_3 - kern_sz_3 + 1) // (stride_3 * pool_sz_3)

        # Conv 4
        self.conv4 = nn.Conv2d(kernels_3, kernels_4, kern_sz_4, padding=pad_4)
        # self.pool4 = nn.MaxPool2d(pool_sz_4, pool_sz_4)

        H_4_pool = (H_3_pool + 2*pad_4 - kern_sz_4 + 1) // (stride_4 * pool_sz_4)
        W_4_pool = (W_3_pool + 2*pad_4 - kern_sz_4 + 1) // (stride_4 * pool_sz_4)

        # Linear Layers
        self.fc1 = nn.Linear(kernels_4 * H_4_pool * W_4_pool, h_5)
        self.fc2 = nn.Linear(h_5, h_6)
        self.fc3 = nn.Linear(h_6, frames_out * H_in * W_in)

    def __call__(self, x):

        # Conv 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Conv 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Conv 3
        x = self.conv3(x)
        x = torch.relu(x)
        # x = self.pool3(x)

        # Conv 4
        x = self.conv4(x)
        x = torch.relu(x)
        # x = self.pool4(x)

        # Linear Layers
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
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

