import torch.nn as nn
import torch
from networks.HO_Nets.HO_Posenet import HO_Posenet
from networks.HO_Nets.HO_VoxelNet import HO_VoxelNet


class HO_PVNet(nn.Module):
    def __init__(self):
        super(HO_PVNet, self).__init__()
        self.poseNet = HO_Posenet(input_channels = 1, hand_channels=21, obj_channels=8)
        self.voxelNet = HO_VoxelNet(input_channels=1+21+8)
        self._initialize_weights()
        self.sigmoid = nn.Sigmoid()

    def forward(self, voxel88, voxel44):
        poseresult = self.poseNet(voxel88)
        in2 = torch.cat((voxel44, poseresult['handpose'], poseresult['objpose']), dim=1)
        out_voxel = self.voxelNet(in2)
        out_voxel = self.sigmoid(out_voxel)
        result = {
            'handpose': poseresult['handpose'],
            'objpose': poseresult['objpose'],
            'voxel': out_voxel
        }
        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
