from config import *
import sys
sys.path.append(PROJECT_PATH)

from dataset.HO_Data.HO_PVNet import *
from dataset.HO_Data.convert import *
from dataset.HO_Data.vis_util import *
from networks.HO_Nets.HO_PVNet import HO_PVNet
import argparse
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--figMode', default='2D',choices=['2D', '3DPC', '3DMesh',''])
parser.add_argument('--multi_GPU', type=bool, default=True,help='If the model to load was trained in multi-GPU')
args = parser.parse_args()
#######################################################################################
###################### Model weights load ###########################
print('==> Constructing model ... HO_PVNet')
pvnet = HO_PVNet()
if(args.multi_GPU):
    pvnet = torch.nn.DataParallel(pvnet)
pvnet = pvnet.to(device, dtype)
pvnet.load_state_dict(torch.load(PVNet_ckpt)['model_state_dict'])


#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/PVNet/'
    if (os.path.exists(saveFolder)):
        shutil.rmtree(saveFolder)
    os.mkdir(saveFolder)

with open(file_path) as tf:
    records = tf.readlines()
    random.shuffle(records)
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

        # voxel88 = transform.voxelize88(pointcloud, refpoint)
        # voxel88 = torch.Tensor(voxel88).unsqueeze(0).to(device, dtype)
        #
        # voxel44 = transform.voxelize44(pointcloud, refpoint)
        # voxel44 = torch.Tensor(voxel44).unsqueeze(0).to(device, dtype)
        joints_world = Main_pixelToworld(gt_hand_uvd.copy(), ux, uy, fx, fy)
        bbox_world = Main_pixelToworld(gt_obj_uvd.copy(), ux, uy, fx, fy)
        handmesh_world = Main_pixelToworld(gt_handMesh_uvd.copy(), ux, uy, fx, fy)
        objmesh_world = Main_pixelToworld(gt_objmesh_uvd.copy(), ux, uy, fx, fy)

        sample = {
            'points': pointcloud,
            'joints': joints_world,
            'bbox': bbox_world,
            'handmesh': handmesh_world,
            'objmesh': objmesh_world,
            'refpoint': refpoint,
        }
        voxel88, heatmap_joints, heatmap_bbox, voxel44, mesh_voxel = transform(sample)
        voxel88 = torch.from_numpy(voxel88.reshape((1,1, *voxel88.shape))).to(dtype)
        voxel44 = torch.from_numpy(voxel44.reshape((1,1, *voxel44.shape))).to(dtype)
        mesh_voxel = torch.from_numpy(mesh_voxel.reshape((1, 1, *mesh_voxel.shape))).to(dtype)
        ####################### get prediction ############################
        with torch.no_grad():
            result = pvnet(voxel88,voxel44)
            out_voxel = result['voxel']
            out_voxel[out_voxel > 0.8] = 1
            out_voxel[out_voxel <= 0.8] = 0
            print('per voxel loss:', torch.nn.BCELoss()(out_voxel.cpu(), mesh_voxel) / np.count_nonzero(mesh_voxel))
            voxel44[out_voxel > 0.8] = 1

            ###################### hand conversion ####################
            hand_hmp = result['handpose'][0].cpu().numpy()
            handUVD, handxyz = pred2Org_handjoints(hand_hmp, refpoint, ux, uy, fx, fy)
            #print('hand points loss:', np.mean(np.linalg.norm((handxyz - annot['handJoints3D']), axis=1)))

            ################# object conversion #######################
            obj_hmp = result['objpose'][0].cpu().numpy()
            objbboxUVD, objxyz = pred2Org_objbbox(obj_hmp, refpoint, ux, uy, fx, fy)
            #print('obj points loss:', np.mean(np.linalg.norm((objxyz - annot['objCorners3D']),axis=1)))

            ################# voxel conversion #######################
            out_voxel = out_voxel[0].cpu().numpy()
            mesh_voxel = mesh_voxel[0].cpu().numpy()
            # print(out_voxel)
            print(np.argwhere(out_voxel[0] != 0).shape)

            meshUVD, _ = pred2Org_mesh(np.argwhere(out_voxel[0]), refpoint, ux, uy, fx, fy)
            fullmesh_uvd, _ = pred2Org_mesh(np.argwhere(mesh_voxel[0]), refpoint, ux, uy, fx, fy)
        # ###################### compare GT & prediction on visualization #################
        if (args.figMode != ''):
            fileName = saveFolder + '/' + folder + '_' + file + '.png'
            fig = plt.figure(figsize=(30, 30))

            if (args.figMode == '2D'):
                ax0 = fig.add_subplot(1, 2, 1)
                plotOnOrgImg(ax0, gt_hand_uvd, gt_obj_uvd, depth)
                ax0.scatter(fullmesh_uvd[:, 0], fullmesh_uvd[:, 1], c='b', s=10)
                ax0.title.set_text('Ground Truth')

                ax1 = fig.add_subplot(1, 2, 2)
                plotOnOrgImg(ax1, handUVD, objbboxUVD, depth)
                ax1.scatter(meshUVD[:, 0], meshUVD[:, 1], c='b', s=10)
                ax1.title.set_text('Prediction')

            elif (args.figMode == '3DPC'):
                ax0 = fig.add_subplot(1, 2, 1, projection='3d')
                ax0.view_init(elev=0, azim=-50)
                ax0.scatter(fullmesh_uvd[:, 0], fullmesh_uvd[:, 1], fullmesh_uvd[:, 2], c=fullmesh_uvd[:, 2], s=30)
                ax0.title.set_text('Ground Truth')

                ax1 = fig.add_subplot(1, 2, 2, projection='3d')
                ax1.view_init(elev=0, azim=-50)
                ax1.scatter(meshUVD[:, 0], meshUVD[:, 1], meshUVD[:, 2], c=meshUVD[:, 2], s=30)
                ax1.title.set_text('Prediction')

            plt.show()
            plt.savefig(fileName)
            plt.close()
