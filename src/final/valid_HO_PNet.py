from config import *
import sys
sys.path.append(PROJECT_PATH)

import torch
from dataset.HO_Data.codelab_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_Posenet import HO_Posenet
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--saveJSON', type=bool, default=False, help='save json or not')
parser.add_argument('--handout', type=str, default='hand_pnet_{}.json'.format(validset), help='File to save the predictions.')
parser.add_argument('--objout', type=str, default='obj_pnet_{}.json'.format(validset), help='File to save the predictions.')
parser.add_argument('--figMode', default='2D',choices=['2D', '3DPC',''])
parser.add_argument('--multi_GPU', type=bool, default=True,help='If the model to load was trained in multi-GPU')
args = parser.parse_args()
#######################################################################################
###################### Model weights load ###########################
print('==> Constructing model ... HO_PNet')
posenet = HO_Posenet(input_channels=1, hand_channels=handpoints_num, obj_channels=objpoints_num)
if(args.multi_GPU):
    posenet = torch.nn.DataParallel(posenet)
posenet = posenet.to(device, dtype)
posenet.load_state_dict(torch.load(PNet_ckpt)['model_state_dict'])

#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)

# init output containers
hand_xyz_list= list()
obj_xyz_list = list()
hand_out_path = PROJECT_PATH+'/'+args.handout
obj_out_path = PROJECT_PATH + '/' + args.objout

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/PNet/'
    if (os.path.exists(saveFolder)):
        shutil.rmtree(saveFolder)
    os.mkdir(saveFolder)

with open(file_path) as tf:
    records = tf.readlines()
    #random.shuffle(records)
    for record in records:
        print(record)
        folder, file = tuple(record.rstrip().split('/'))
        depthpath = os.path.join(folder_path, folder, 'depth', file + '.png')
        annotpath = os.path.join(folder_path, folder, 'meta', file + '.pkl')
        depth = read_depth_img(depthpath)
        annot = np.load(annotpath, allow_pickle=True)
        camMat = annot['camMat']
        fx = camMat[0, 0]
        fy = camMat[1, 1]
        ux = camMat[0, 2]
        uy = camMat[1, 2]

        if (validset != 'evaluation'):
            handJoints = annot['handJoints3D']
            handJoints = handJoints[jointsMapManoToSimple]
            objCorners = annot['objCorners3D']
            gt_hand_uvd = project_3D_points(camMat, handJoints)
            gt_obj_uvd = project_3D_points(camMat, objCorners)
            ################ get the common center point of hand and object ###########
            objcenter = np.mean(gt_obj_uvd, axis=0)
            com = np.mean(np.array([gt_hand_uvd[0], objcenter]), axis=0)
        else:
            ################ for evaluation set ############
            objCorners = annot['objCorners3D']
            obj_uvd = project_3D_points(camMat, objCorners)
            handroot = np.array(annot['handJoints3D'])
            handroot_uvd = project_3D_points(camMat, handroot.reshape(1, -1))
            handroot_uvd = handroot_uvd[0]
            ################ get the common center point of hand and object ###########
            objcenter = np.mean(obj_uvd, axis=0)
            com = np.mean(np.array([handroot_uvd, objcenter]), axis=0)

        #################### v2v approach : voxel segment and generate heatmap ###############
        pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
        pointcloud = pointcloud.reshape(-1, 3)
        refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
        refpoint = np.array(refpoint)

        ################ below part is old. needed if voxelization is to be checked from respective file ###########
        # joints_world = Main_pixelToworld(handJoints_uvd.copy(), ux, uy, fx, fy)
        # bbox_world = Main_pixelToworld(obj_uvd.copy(), ux, uy, fx, fy)
        #
        # sample = {
        #     'points': pointcloud,
        #     'joints': joints_world,
        #     'bbox': bbox_world,
        #     'refpoint': refpoint,
        # }
        # voxel88, heatmap_joints, heatmap_bbox = transform(sample)
        # testVis(depthvoxel, heatmap_joints, heatmap_bbox)
        # voxel88 = torch.from_numpy(voxel88.reshape((1, 1, *voxel88.shape))).to(device, dtype)

        voxel88 = transform.voxelize88(pointcloud, refpoint)
        voxel88 = torch.Tensor(voxel88).unsqueeze(0).to(device, dtype)
        ####################### get prediction ############################
        with torch.no_grad():
            poseResult = posenet(voxel88)

            ##################### post processing of outputs ################
            ###################### hand conversion ####################
            hand_hmp = poseResult['handpose'][0].cpu().numpy()
            handUVD, handxyz = pred2Org_handjoints(hand_hmp, refpoint, ux, uy, fx, fy)


            ################# object conversion #######################
            obj_hmp = poseResult['objpose'][0].cpu().numpy()
            objbboxUVD, objxyz = pred2Org_objbbox(obj_hmp, refpoint, ux, uy, fx, fy)


            if (validset != 'evaluation'):
                print('hand points loss:', np.mean(np.linalg.norm((handxyz - annot['handJoints3D']), axis=1)))
                print('obj points loss:', np.mean(np.linalg.norm((objxyz - annot['objCorners3D']), axis=1)))

            # ###################### compare GT & prediction on visualization #################
            if (args.figMode != ''):
                fileName = saveFolder + '/' + folder + '_' + file + '.png'
                fig = plt.figure(figsize=(30, 30))
                if (validset == 'evaluation'):
                    ax1 = fig.add_subplot(1, 1, 1)                    
                    ax1.title.set_text('Prediction')
                    if (args.figMode == '2D'):
                        plotOnOrgImg(ax1, handUVD, objbboxUVD, depth)
                    elif (args.figMode == '3DPC'):
                        draw3dpose(ax1, handUVD, objbboxUVD)
                else:
                    if (args.figMode == '2D'):
                        ax0 = fig.add_subplot(1, 2, 1)
                        plotOnOrgImg(ax0,gt_hand_uvd, gt_obj_uvd, depth)
                        ax0.title.set_text('Ground Truth')

                        # show Prediction
                        ax1 = fig.add_subplot(1, 2, 2)
                        plotOnOrgImg(ax1, handUVD, objbboxUVD, depth)
                        ax1.title.set_text('Prediction')

                    elif (args.figMode == '3DPC'):
                        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
                        ax0.view_init(elev=0, azim=-50)
                        draw3dpose(ax0, gt_hand_uvd, gt_obj_uvd)
                        ax0.title.set_text('Ground Truth')

                        ##### show Prediction
                        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
                        ax1.view_init(elev=0, azim=-50)
                        draw3dpose(ax1, handUVD, objbboxUVD)
                        ax1.title.set_text('Prediction')

                plt.savefig(fileName)
                plt.close()


            if (args.saveJSON):
                hand_xyz_list.append(handxyz)
                obj_xyz_list.append(objxyz)

    ########## dump results. During testing of qualitative result we don't want to dump ###################
    if(args.saveJSON):
        dump(hand_out_path, hand_xyz_list, [])
        dump(obj_out_path, obj_xyz_list, [])
