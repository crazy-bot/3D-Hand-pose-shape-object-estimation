from config import *
import sys
sys.path.append(PROJECT_PATH)

from dataset.HO_Data.HO_voxel_test import *
from dataset.HO_Data.vis_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_VoxelNet import HO_VoxelNet
import argparse
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--figMode', default='2D',choices=['2D', '3DPC', '3DMesh',''])
parser.add_argument('--multi_GPU', type=bool, default=True,help='If the model to load was trained in multi-GPU')
args = parser.parse_args()

#######################################################################################
###################### Model weights load ###########################
print('==> Constructing model ... HO_VNet')
vnet = HO_VoxelNet(input_channels=1+21+8)
if(args.multi_GPU):
    vnet = torch.nn.DataParallel(vnet)
vnet = vnet.to(device, dtype)
vnet.load_state_dict(torch.load(VNet_ckpt)['model_state_dict'])

#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/VNet/'
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

        handJoints = annot['handJoints3D']
        handJoints = handJoints[jointsMapManoToSimple]
        objCorners = annot['objCorners3D']
        gt_hand_uvd = project_3D_points(camMat, handJoints)
        gt_obj_uvd = project_3D_points(camMat, objCorners)

        _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])
        gt_handMesh_uvd = project_3D_points(camMat, handMesh)
        objMesh = read_obj(
            os.path.join(OBJ_MODEL_PATH, annot['objName'], 'textured_2358.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']
        gt_objmesh_uvd = project_3D_points(camMat, objMesh.v)
        fullmesh_uvd = np.concatenate([gt_handMesh_uvd, gt_objmesh_uvd], axis=0)
        ################ get the common center point of hand and object ###########
        objcenter = np.mean(gt_obj_uvd, axis=0)
        com = np.mean(np.array([gt_hand_uvd[0], objcenter]), axis=0)

        #################### v2v approach : voxel segment and generate heatmap ###############
        pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
        pointcloud = pointcloud.reshape(-1, 3)

        h, w = depth.shape
        refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
        refpoint = np.array(refpoint)
        joints_world = Main_pixelToworld(gt_hand_uvd.copy(), ux, uy, fx, fy)
        bbox_world = Main_pixelToworld(gt_obj_uvd.copy(), ux, uy, fx, fy)
        fullmesh_world = Main_pixelToworld(fullmesh_uvd.copy(), ux, uy, fx, fy)

        sample = {
            'points': pointcloud,
            'joints': joints_world,
            'bbox': bbox_world,
            'fullmesh': fullmesh_world,
            'refpoint': refpoint,
        }
        depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel = transform(sample)
        fullmesh_uvd, _ = pred2Org_mesh(np.argwhere(mesh_voxel), refpoint, ux, uy, fx, fy)

        fig = plt.figure(figsize=(30, 30))
        ax0 = fig.add_subplot(1, 1, 1)
        ax0.imshow(depth, 'gray')
        ax0.scatter(fullmesh_uvd[:, 0], fullmesh_uvd[:, 1], c='b', s=10)
        ax0.title.set_text('Ground Truth')
        #plt.show()
        #testVis(mesh_voxel, heatmap_joints, heatmap_bbox)
        ###################### get prediction ############################
        depth_voxel = torch.from_numpy(depth_voxel.reshape((1, 1, *depth_voxel.shape))).to(device, dtype)
        #mesh_voxel = torch.from_numpy(mesh_voxel.reshape((1,1, *mesh_voxel.shape))).to(device, dtype)
        heatmap_joints = torch.from_numpy(heatmap_joints.reshape(1, *heatmap_joints.shape)).to(device, dtype)
        heatmap_bbox = torch.from_numpy(heatmap_bbox.reshape(1, *heatmap_bbox.shape)).to(device, dtype)
        inputs = torch.cat((depth_voxel, heatmap_joints, heatmap_bbox), dim=1)

        with torch.no_grad():
            out_voxel  = vnet(inputs)
            ################# voxel conversion #######################
            out_voxel = out_voxel[0].cpu().numpy()
            out_voxel[out_voxel > 0.8] = 1
            out_voxel[out_voxel <= 0.8] = 0
            #print(out_voxel)
            print(np.argwhere(out_voxel[0] != 0).shape)
            meshUVD, _ = pred2Org_mesh(np.argwhere(out_voxel[0]), refpoint, ux, uy, fx, fy)

        # ###################### compare GT & prediction on visualization #################
        if (args.figMode != ''):
            fileName = saveFolder + '/' + folder + '_' + file + '.png'
            fig = plt.figure(figsize=(30, 30))

            if (args.figMode == '2D'):
                ax0 = fig.add_subplot(1, 2, 1)
                ax0.imshow(depth, 'gray')
                ax0.scatter(fullmesh_uvd[:, 0], fullmesh_uvd[:, 1], c='g', s=10)
                ax0.title.set_text('Ground Truth')


                ax1 = fig.add_subplot(1, 2, 2)
                ax1.imshow(depth, 'gray')
                ax1.scatter(meshUVD[:, 0], meshUVD[:, 1], c='g', s=10)
                ax1.title.set_text('Prediction')

            elif (args.figMode == '3DPC'):
                ax0 = fig.add_subplot(1, 2, 1, projection='3d')
                #ax0.scatter(fullmesh_uvd[:, 0], fullmesh_uvd[:, 1], fullmesh_uvd[:, 2], c=fullmesh_uvd[:, 2], s=30)
                ax0.title.set_text('Ground Truth')
                ax0.axis('off')
                ax0.voxels(mesh_voxel)

                ax1 = fig.add_subplot(1, 2, 2, projection='3d')
                #ax1.scatter(meshUVD[:, 0], meshUVD[:, 1], meshUVD[:, 2], c=meshUVD[:, 2], s=30)
                ax1.voxels(mesh_voxel)
                ax1.title.set_text('Prediction')
        #plt.show()
        plt.savefig(fileName)
        plt.close()

