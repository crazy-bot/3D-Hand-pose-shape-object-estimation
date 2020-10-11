from config import *
import sys
sys.path.append(PROJECT_PATH)

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from train_helper import train_HO_SNet_v1, val_HO_SNet_v1
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet
from dataset.HO_Data.HO_shape_v4 import Ho3DDataset


ap = argparse.ArgumentParser()
ap.add_argument("--load_ckpt", type=str, help="relative path to load checkpoint", default='')
ap.add_argument("--save_ckpt", type=str, help="relative path to save checkpoint", default='HO_SNet_v4')
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
val_set = Ho3DDataset(root=DATA_DIR,pathFile='valid.txt',augmentation=False,dtype=dtype, isValid=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
print('train size--', len(train_loader))
print('val size--', len(val_loader))

#######################################################################################
###################### Model, criterion and optimizer ###########################
print('==> Constructing model ... HO_SNet_v4')
net = HO_ShapeNet(input_channels=1,hand_channels=no_handverts,obj_channels=no_objverts)

if(args.multiGPU):
    net = nn.DataParallel(net)
net = net.to(device, dtype)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15,20,25], gamma=0.5, last_epoch=-1)
#######################################################################################
################ load predefined checkpoint ###############
if args.load_ckpt is not '':
    checkpoint_file = os.path.join(CHECKPOINT_BASE, args.load_ckpt)
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of found'

    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


#######################################################################################
##################### Train and Validate #################
print('==> Training ..')
for epoch in range(start_epoch, start_epoch + epochs_num):
    print('Epoch: {}'.format(epoch))
    train_HO_SNet_v1(net, criterion, optimizer, train_loader, device=device)
    val_HO_SNet_v1(net, val_loader, device=device)
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