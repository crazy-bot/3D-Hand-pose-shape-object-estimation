import torch.nn as nn
import torch.nn.functional as F
import torch


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Inception3DBlock(nn.Module):
    def __init__(self, in_planes, f1,f3,f5):
        super(Inception3DBlock, self).__init__()
        self.conv1= nn.Sequential(
            nn.Conv3d(in_planes, f1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(f1),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_planes, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(f3),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_planes, f5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(f5),
            nn.ReLU(True)
        )

    def forward(self, X):
        out = torch.cat((self.conv1(X),self.conv3(X),self.conv5(X)),dim=1)
        return F.relu(out, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class HO_ShapeNet(nn.Module):
    def __init__(self, input_channels, hand_channels, obj_channels):
        super(HO_ShapeNet, self).__init__()
        self.hand_channels = hand_channels
        self.obj_channels = obj_channels

        self.front_layers = nn.Sequential(
            Inception3DBlock(input_channels,16,32,64), # (b,112,44,44,44)
            Pool3DBlock(2),                            # (b,112,22,22,22)
            Inception3DBlock(16 + 32 + 64, 8, 16, 32),  # (b,56,22,22,22)
            Pool3DBlock(2),                             # (b,56,11,11,11)
            Inception3DBlock(8+16+32, 4, 8, 16),        # (b,28,11,11,11)
            Inception3DBlock(4+8+16 , 2,4,8),  # (b,14,11,11,11)
        )
        self.handlayer_mid = Basic3DBlock(14, 7, 1)
        self.handlayer_end = nn.Linear(7 * 11 * 11 * 11 ,hand_channels*3)

        self.objlayer_mid = Basic3DBlock(14, 12, 1)
        self.objlayer_end = nn.Linear(12 * 11 * 11 * 11, obj_channels*3)

        self._initialize_weights()

    def forward(self, x):
        batch_sz = x.size()[0]
        x_front = self.front_layers(x) #  # #(b,14,11,11,11)

        in_hand = self.handlayer_mid(x_front)
        hand_verts = self.handlayer_end(in_hand.view(batch_sz,-1))

        in_obj = self.objlayer_mid(x_front)
        obj_verts = self.objlayer_end(in_obj.view(batch_sz,-1))

        hand_verts = hand_verts.reshape((batch_sz,self.hand_channels,3))
        obj_verts = obj_verts.reshape((batch_sz, self.obj_channels, 3))
        result = {
            'handverts': hand_verts,
            'objverts': obj_verts
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
