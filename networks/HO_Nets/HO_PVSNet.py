import torch.nn as nn
import torch
from networks.HO_Nets.HO_Posenet import HO_Posenet
from networks.HO_Nets.HO_VoxelNet import HO_VoxelNet
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet


class HO_PVSNet(nn.Module):
    def __init__(self):
        super(HO_PVSNet, self).__init__()
        self.posenet = HO_Posenet(input_channels = 1, hand_channels=21, obj_channels=8)
        self.voxelNet = HO_VoxelNet(input_channels=1+21+8)
        self.shapeNet = HO_ShapeNet(input_channels = 1+21+8, hand_channels=778,obj_channels=2358)
        #self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, voxel88, voxel44):
        poseresult = self.posenet(voxel88)
        in2 = torch.cat((voxel44, poseresult['handpose'],poseresult['objpose']), dim=1)
        out_voxel = self.voxelNet(in2)
        out_voxel = nn.Sigmoid()(out_voxel)
        voxel44[out_voxel>0.8] = 1
        in3 = torch.cat((voxel44, poseresult['handpose'], poseresult['objpose']), dim=1)
        shaperesult = self.shapeNet(in3)
        result = {
            'handpose': poseresult['handpose'],
            'objpose': poseresult['objpose'],
            'voxel': out_voxel,
            'handverts': shaperesult['handverts'],
            'objverts': shaperesult['objverts']
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
