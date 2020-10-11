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

    
class MaxPool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)

class AvgPool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(AvgPool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                               output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                               output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder_res1 = Inception3DBlock(30, 8,16,32)
        self.encoder_res2 = Inception3DBlock(8+16+32, 16,32,64)

        self.mid_res = Inception3DBlock(16+32+64, 16,32,64)

        self.decoder_res2 = Inception3DBlock(16+32+64, 8,16,32)
        self.decoder_res1 = Inception3DBlock(8+16+32, 4,10,16)

        self.skip_res1 = Inception3DBlock(30, 8, 16, 32)
        self.skip_res2 = Inception3DBlock(8+16+32, 16,32,64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)     # (b,56,44,44,44)
        x = self.encoder_res1(x)        # (b,56,44,44,44)
        x,indices0 = F.max_pool3d(x,kernel_size=2, stride=2, return_indices=True)   # (b,56,22,22,22)
        dim_1 = x.size()
        skip_x2 = self.skip_res2(x) # (b,112,22,22,22)
        x = self.encoder_res2(x)    # (b,112,22,22,22)
        x,indices1 = F.max_pool3d(x,kernel_size=2, stride=2, return_indices=True)   # (b,112,11,11,11)
        dim_2 = x.size()

        x = self.mid_res(x)         # (b,112,11,11,11)

        x = F.max_unpool3d(x, indices1, kernel_size=2, stride=2)         # (b,112,22,22,22)
        x = x + skip_x2                                                  # (b,112,22,22,22)
        x = self.decoder_res2(x)                                        # (b,56,22,22,22)

        x = F.max_unpool3d(x, indices0, kernel_size=2, stride=2)        # (b,56,44,44,44)
        x = x + skip_x1                                                  # (b,56,44,44,44)
        x = self.decoder_res1(x)                                        # (b,22,44,44,44)

        return x


class HO_VoxelNet(nn.Module):
    def __init__(self, input_channels,):
        super(HO_VoxelNet, self).__init__()

        self.voxnet = nn.Sequential(
            EncoderDecoder(),                           # (b,30,44,44,44)
            Basic3DBlock(input_channels, 11, 3),                     # (b,7,44,44,44)
            Basic3DBlock(11, 1, 1),                      # (b,1,44,44,44)
            #nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        ######### voxel of vertices ###########
        out_voxel = self.voxnet(x)                       # (b,1,44,44,44)
        return out_voxel

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
