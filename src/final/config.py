import torch

################# path of data and others #################
PROJECT_PATH = '/handpose/suparna/Honnnotate-3D-Pose-Mesh'
MANO_MODEL_PATH = PROJECT_PATH+'/mano/models/MANO_RIGHT.pkl'
OBJ_MODEL_PATH = '/data/Guha/Honnotate/YCB_Video_Models/models/'
DATA_DIR = '/data/Guha/Honnotate/HO3D/'
CHECKPOINT_BASE = r'/data/Guha/Honnotate/checkpoint/best_models/A2/'
#CHECKPOINT_BASE = r'/handpose/suparna/thesis_ckpt/'
SAVEFIG_DIR = PROJECT_PATH+'/output'

################## Training parameters #######################
save_checkpoint = True #if checkpoint to be saved
checkpoint_per_epochs = 1 #intervals of epochs to save checkpoint
start_epoch = 0 
epochs_num = 40 # training will be performed from start_epoch to start_epoch + epochs_num
batch_size = 20

########### path of all pretrained checkpoints ##################
PNet_ckpt= CHECKPOINT_BASE + '/HO_PNet/epoch48.pth'
SNet_ckpt = CHECKPOINT_BASE + '/HO_SNet_v6/epoch54.pth'
VNet_ckpt = CHECKPOINT_BASE + '/HO_VNet/epoch37.pth'
PVNet_ckpt = CHECKPOINT_BASE + '/HO_PVNet/epoch14.pth'
PVSNet_ckpt = CHECKPOINT_BASE + '/HO_PVSNet_v6.2/epoch27.pth'

############ Config for Validation files {validset can be 'valid','evaluation'} ##############
validset = 'valid'

############## Evaluation file config {evalset can be 'valid','evaluation'} ###########
evalset = 'evaluation'
objJson_file = 'obj_pvsnet_v6_evaluation.json'
handJson_file = 'pip_hand_PSNetv2_valid.json'
EvalDir = PROJECT_PATH+'/{}'
objJsonPath = EvalDir.format(objJson_file)
handJsonPath = EvalDir.format(handJson_file)


########### Do not change!! ##################
handpoints_num = 21
objpoints_num = 8
no_handverts = 778
no_objverts = 2358
cubic_size = 200
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float
