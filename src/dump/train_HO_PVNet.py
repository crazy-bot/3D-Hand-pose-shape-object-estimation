from config import *
import sys
sys.path.append(PROJECT_PATH)

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from src.train_helper import train_HO_PVNet, val_HO_PVNet
from networks.HO_Nets.HO_PVNet import HO_PVNet
from dataset.HO_Data.HO_PVNet import Ho3DDataset


ap = argparse.ArgumentParser()
ap.add_argument("--load_ckpt", type=str, help="relative path to load checkpoint", default='')
ap.add_argument("--save_ckpt", type=str, help="relative path to save checkpoint", default='HO_PVNet')
ap.add_argument("--multiGPU", type=bool, help="if multi GPU training needed", default=True)

args = ap.parse_args()
checkpoint_dir = os.path.join(CHECKPOINT_BASE, args.save_ckpt)
if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

#######################################################################################
################# Dataset and data loader ###############
print('==> Preparing data ..')


################# Dataset and loader. 80% training and 20% val set ###############
torch.manual_seed(1)
torch.cuda.manual_seed(1)

train_set = Ho3DDataset(root=DATA_DIR,pathFile='train.txt',augmentation=False,dtype=dtype)
val_set = Ho3DDataset(root=DATA_DIR,pathFile='valid.txt',augmentation=False,dtype=dtype,isValid=False)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
print('train size--', len(train_loader))
print('val size--', len(val_loader))

#######################################################################################
###################### Model, criterion and optimizer ###########################
print('==> Constructing model .. HO_PVNet')
net = HO_PVNet()

if(args.multiGPU):
    net = nn.DataParallel(net)
net = net.to(device, dtype)

criterion1 = nn.MSELoss()
# weight = torch.Tensor([(44*44*44) / (no_handverts+no_objverts)*2]).to(device,dtype)
# criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=weight,reduction='none') # -log(sigmoid(output))
#criterion2 = PixelWiseCrossEntropyLoss()

# optimizer = optim.Adam(net.module.poseNet.parameters(),lr=0.0001,amsgrad=True)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15], gamma=0.5, last_epoch=-1)

optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15,20,25], gamma=0.3, last_epoch=-1)

############### freeze specific layers, if needed ###############
for param in net.module.poseNet.parameters():
    param.requires_grad = False

#######################################################################################
################ load predefined checkpoint of this network if to be resumed training ###############
if args.load_ckpt is not '':
    checkpoint_file = os.path.join(CHECKPOINT_BASE, args.load_ckpt)
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of found'

    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    ############### load all the checkpoint weights #################
    hops_weights = torch.load(PSNet_ckpt)['model_state_dict']
    hops_weights = {k.replace('module.', ''): hops_weights[k] for k in list(hops_weights.keys())}
    posenet_weights = {k.replace('posenet.', ''): hops_weights[k] for k in list(hops_weights.keys()) if 'posenet' in k}

    # old_weights = torch.load(PNet_ckpt)['model_state_dict']
    # new_weights = {k.replace('module.',''):old_weights[k] for k in list(old_weights.keys()) }
    net.module.poseNet.load_state_dict(posenet_weights)

    old_weights = torch.load(VNet_ckpt)['model_state_dict']
    new_weights = {k.replace('module.', ''): old_weights[k] for k in list(old_weights.keys())}
    net.module.voxelNet.load_state_dict(new_weights)
    print('initial weights loaded')


#######################################################################################
##################### Train and Validate #################
print('==> Training ..')
for epoch in range(start_epoch, start_epoch + epochs_num):
    print('Epoch: {}'.format(epoch))
    train_HO_PVNet(net, optimizer, train_loader, device=device)
    val_HO_PVNet(net, val_loader, device=device)
    ########### take schedular step #########
    scheduler.step()

    if save_checkpoint and epoch % checkpoint_per_epochs == 0:
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)