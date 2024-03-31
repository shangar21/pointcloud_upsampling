import torch
import torch.nn as nn
import sys
sys.path.append('./pvcnn')
from modules import pvconv
from models.utils import create_pointnet_components, create_mlp_components

class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

        self.channels_point = channels_point
        self.concat_channels_point = concat_channels_point
        
        self.point_features = nn.ModuleList(layers)



    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return torch.cat(out_features_list, dim=1) 

class BoltNet(nn.Module):
    def __init__(self, num_classes, num_shapes, out_points=2048):
        super(BoltNet, self).__init__()
        self.encoder = PVCNN(num_classes, num_shapes)
        num_shapes = self.encoder.num_shapes
        channels_point = self.encoder.channels_point
        concat_channels_point = self.encoder.concat_channels_point
        self.decoder, _ = create_mlp_components(in_channels=3 * (num_shapes + channels_point + concat_channels_point),
                                          out_channels=[out_points, 0.2, out_points * 2, 0.2, out_points * 3],
                                          classifier=True, dim=2, width_multiplier=1)
        self.decoder = nn.Sequential(*self.decoder)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.encoder(x) 
        x = x.reshape(x.size(0), 3 * (self.encoder.num_shapes + self.encoder.channels_point + self.encoder.concat_channels_point), 1)
        x = self.decoder(x)
        x = x.reshape(x.size(0), -1, 3)
        return x
