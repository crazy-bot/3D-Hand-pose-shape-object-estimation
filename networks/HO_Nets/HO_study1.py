import torch.nn as nn
import torch
from networks.HO_Nets.HO_Posenet import HO_Posenet
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet


class HO_PSNet(nn.Module):
    def __init__(self):
        super(HO_PSNet, self).__init__()
        self.posenet = HO_Posenet(input_channels = 1, hand_channels=21, obj_channels=8)
        self.shapeNet = HO_ShapeNet(input_channels = 1+21+8, hand_channels=778,obj_channels=2358)
        self._initialize_weights()

    def forward(self, voxel88, voxel44):
        result = self.posenet(voxel88)
        in2 = torch.cat((voxel44, result['handpose'],result['objpose']), dim=1)
        shaperesult = self.shapeNet(in2)
        result = {
            'handpose': result['handpose'],
            'objpose': result['objpose'],
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