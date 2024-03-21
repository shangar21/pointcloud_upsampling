import torch
import torch.nn as nn
import pytorch3d

class BoltNet(nn.Module):
    def __init__(self, feature_input_size, point_cloud_size, in_channels=1, out_channels=32):
        super(BoltNet, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1) 

    def forward(self, x, normals):
        x = self.conv3d(x)
        return x
