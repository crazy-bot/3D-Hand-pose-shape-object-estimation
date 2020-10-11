from config import *
import sys
sys.path.append(PROJECT_PATH)

from dataset.HO_Data.HO_shape_v2 import *
from dataset.HO_Data.vis_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet
import numpy as np
import argparse
import  shutil


parser = argparse.ArgumentParser()
parser.add_argument('--saveJSON', type=bool, default=False, help='save json or not')
parser.add_argument('--handout', type=str, default='hand_snetv2_valid.json', help='File to save the predictions.')
parser.add_argument('--objout', type=str, default='obj_snetv2_valid.json', help='File to save the predictions.')
parser.add_argument('--figMode', default='2D',choices=['2D', '3DPC', '3DMesh',''])
parser.add_argument('--multi_GPU', type=bool, default=True,help='If the model to load was trained in multi-GPU')
args = parser.parse_args()
#######################################################################################
###################### Model weights load ###########################
print('==> Constructing model ... HO_SNet_v2')
snetv2 = HO_ShapeNet(input_channels=1 + 21 + 8, hand_channels=no_handverts, obj_channels=no_objverts)
if(args.multi_GPU):
    snetv2 = torch.nn.DataParallel(snetv2)
snetv2 = snetv2.to(device, dtype)
snetv2.load_state_dict(torch.load(SNet_v2_ckpt)['model_state_dict'])

#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)

# init output containers
hand_verts_list = list()
obj_verts_list = list()
hand_out_path = PROJECT_PATH + '/' + args.handout
obj_out_path = PROJECT_PATH + '/' + args.objout

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/SNet_v2/'
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

        objMesh = read_obj(
            os.path.join(OBJ_MODEL_PATH, annot['objName'], 'textured_2358.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']
        gt_objmesh_uvd = project_3D_points(camMat, objMesh.v)

        handJoints = annot['handJoints3D']
        handJoints = handJoints[jointsMapManoToSimple]
        objCorners = annot['objCorners3D']
        _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])
        gt_hand_uvd = project_3D_points(camMat, handJoints)
        gt_obj_uvd = project_3D_points(camMat, objCorners)
        gt_handMesh_uvd = project_3D_points(camMat, handMesh)
        ################ get the common center point of hand and object ###########
        objcenter = np.mean(gt_obj_uvd, axis=0)
        com = np.mean(np.array([gt_hand_uvd[0], objcenter]), axis=0)

        #################### v2v approach : voxel segment and generate heatmap ###############
        pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
        pointcloud = pointcloud.reshape(-1, 3)
        refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
        refpoint = np.array(refpoint)

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
        depth_voxel, heatmap_joints, heatmap_bbox, norm_handmesh, norm_objmesh = transform.train_transform(sample)
        # testVis(depth_voxel, joints, objCorners,norm_handmesh, norm_objmesh)

        depth_voxel = torch.from_numpy(depth_voxel.reshape((1, 1, *depth_voxel.shape))).to(device, dtype)
        heatmap_joints = torch.from_numpy(heatmap_joints.reshape(1, *heatmap_joints.shape)).to(device, dtype)
        heatmap_bbox = torch.from_numpy(heatmap_bbox.reshape(1, *heatmap_bbox.shape)).to(device, dtype)
        inputs = torch.cat((depth_voxel, heatmap_joints, heatmap_bbox), dim=1)

        ####################### get prediction ############################
        with torch.no_grad():
            result = snetv2(inputs)

            ##################### post processing of outputs ################
            ###################### hand conversion ####################
            handmesh = result['handverts'][0].cpu().numpy()
            handmeshUVD, handverts = pred2Org_mesh(handmesh, refpoint, ux, uy, fx, fy)
            print('hand mesh loss:', np.mean(np.linalg.norm((handverts - handMesh),axis=1)))

            ################# object conversion #######################
            objmesh = result['objverts'][0].cpu().numpy()
            objmeshUVD, objverts = pred2Org_mesh(objmesh, refpoint, ux, uy, fx, fy)
            print('obj mesh loss:', np.mean(np.linalg.norm((objverts - objMesh.v),axis=1)))

            ######################### compare GT & prediction on visualization #################
        if (args.figMode != ''):
            fileName = saveFolder + '/' + folder + '_' + file + '.png'
            fig = plt.figure(figsize=(30, 30))

            if (args.figMode == '2D'):
                ax0 = fig.add_subplot(1, 2, 1)
                ax0.imshow(depth, 'gray')
                ax0.scatter(gt_handMesh_uvd[:, 0], gt_handMesh_uvd[:, 1], c=gt_handMesh_uvd[:, 2], s=15)
                ax0.scatter(gt_objmesh_uvd[:, 0], gt_objmesh_uvd[:, 1], c=gt_objmesh_uvd[:, 2], s=15)
                ax0.title.set_text('Ground Truth')

                # show Prediction
                ax1 = fig.add_subplot(1, 2, 2)
                ax1.imshow(depth, 'gray')
                ax1.scatter(handmeshUVD[:, 0], handmeshUVD[:, 1], c=handmeshUVD[:, 2], s=15)
                ax1.scatter(objmeshUVD[:, 0], objmeshUVD[:, 1], c=objmeshUVD[:, 2], s=15)
                ax1.title.set_text('Prediction')

            elif (args.figMode == '3DPC'):
                ax0 = fig.add_subplot(1, 2, 1, projection='3d')
                ax0.view_init(elev=0, azim=-50)
                ax0.scatter(gt_handMesh_uvd[:, 0], gt_handMesh_uvd[:, 1], gt_handMesh_uvd[:, 2],
                            c=gt_handMesh_uvd[:, 2], s=15)
                ax0.scatter(gt_objmesh_uvd[:, 0], gt_objmesh_uvd[:, 1], gt_objmesh_uvd[:, 2],
                            c=gt_objmesh_uvd[:, 2], s=15)
                ax0.title.set_text('Ground Truth')

                ax1 = fig.add_subplot(1, 2, 2, projection='3d')
                ax1.view_init(elev=0, azim=-50)
                ax1.scatter(handmeshUVD[:, 0], handmeshUVD[:, 1], handmeshUVD[:, 2], c=handmeshUVD[:, 2], s=15)
                ax1.scatter(objmeshUVD[:, 0], objmeshUVD[:, 1], objmeshUVD[:, 2], c=objmeshUVD[:, 2], s=15)
                ax1.title.set_text('Prediction')

            elif (args.figMode == '3DMesh'):
                ax0 = fig.add_subplot(1, 2, 1, projection='3d')
                ax0.view_init(elev=0, azim=-50)
                plot3dVisualize(ax0, handMesh, verts=handmeshUVD, flip_x=False, isOpenGLCoords=False, c="r")
                plot3dVisualize(ax0, objMesh, verts=objmeshUVD, flip_x=False, isOpenGLCoords=False, c="b")
                ax0.title.set_text('Ground Truth')

                ax1 = fig.add_subplot(1, 2, 2, projection='3d')
                ax1.view_init(elev=0, azim=-50)
                plot3dVisualize(ax1, handMesh, verts=handmeshUVD, flip_x=False, isOpenGLCoords=False, c="r")
                plot3dVisualize(ax1, objMesh, verts=objmeshUVD, flip_x=False, isOpenGLCoords=False, c="b")
                ax1.title.set_text('Prediction')
            plt.savefig(fileName)
            plt.close()

        if (args.saveJSON):
            hand_verts_list.append(handverts)
            obj_verts_list.append(objverts)

########## dump results. During testing of qualitative result we don't want to dump ###################
if (args.saveJSON):
    dump(hand_out_path, [], hand_verts_list)
    dump(obj_out_path, [], obj_verts_list)
