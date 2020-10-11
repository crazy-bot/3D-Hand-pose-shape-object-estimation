from config import *
import sys
sys.path.append(PROJECT_PATH)

import torch
from dataset.HO_Data.codelab_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet
from networks.HO_Nets.HO_Posenet import HO_Posenet
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--saveJSON', type=bool, default=False, help='save json or not')
parser.add_argument('--handout', type=str, default='hand_baseline_{}.json'.format(validset), help='File to save the predictions.')
parser.add_argument('--objout', type=str, default='obj_baseline_{}.json'.format(validset), help='File to save the predictions.')
parser.add_argument('--figMode', default='3DPC',choices=['2D', '3DPC', '3DMesh',''])
parser.add_argument('--multi_GPU', type=bool, default=True,help='If the model to load was trained in multi-GPU')
args = parser.parse_args()
#######################################################################################
###################### Model weights load  ###########################
print('==> Constructing model ..HO_Posenet')
posenet = HO_Posenet(input_channels=1, hand_channels=handpoints_num, obj_channels=objpoints_num)

print('==> Constructing model ..HO_ShapeNet_v1')
shapenet = HO_ShapeNet(input_channels=1, hand_channels=no_handverts, obj_channels=no_objverts)

if(args.multi_GPU):
    posenet = torch.nn.DataParallel(posenet)
    shapenet = torch.nn.DataParallel(shapenet)

posenet = posenet.to(device, dtype)
shapenet = shapenet.to(device, dtype)
posenet.load_state_dict(torch.load(PNet_ckpt)['model_state_dict'])
shapenet.load_state_dict(torch.load(SNet_v1_ckpt)['model_state_dict'])

#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)

# init output containers
hand_xyz_list, hand_verts_list = list(), list()
obj_xyz_list, obj_verts_list = list(), list()
hand_out_path = PROJECT_PATH+'/'+args.handout
obj_out_path = PROJECT_PATH + '/' + args.objout

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/Baseline/'
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

        if (validset != 'evaluation'):
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

        voxel88 = transform.voxelize88(pointcloud, refpoint)
        voxel88 = torch.Tensor(voxel88).unsqueeze(0).to(device, dtype)

        voxel44 = transform.voxelize44(pointcloud, refpoint)
        voxel44 = torch.Tensor(voxel44).unsqueeze(0).to(device, dtype)

        ####################### get prediction ############################
        with torch.no_grad():
            poseResult = posenet(voxel88)
            shapeResult = shapenet(voxel44)

        ##################### post processing of outputs ################
            ###################### hand conversion ####################
            hand_hmp = poseResult['handpose'][0].cpu().numpy()
            handUVD, handxyz = pred2Org_handjoints(hand_hmp, refpoint, ux, uy, fx, fy)


            handmesh = shapeResult['handverts'][0].cpu().numpy()
            handmeshUVD, handverts = pred2Org_mesh(handmesh, refpoint, ux, uy, fx, fy)


            ################# object conversion #######################
            obj_hmp = poseResult['objpose'][0].cpu().numpy()
            objbboxUVD, objxyz = pred2Org_objbbox(obj_hmp, refpoint, ux, uy, fx, fy)


            objmesh = shapeResult['objverts'][0].cpu().numpy()
            objmeshUVD, objverts = pred2Org_mesh(objmesh, refpoint, ux, uy, fx, fy)


            if (validset != 'evaluation'):
                print('hand points loss:', np.mean(np.linalg.norm((handxyz - annot['handJoints3D']), axis=1)))
                print('hand mesh loss:', np.mean(np.linalg.norm((handverts - handMesh), axis=1)))
                print('obj points loss:', np.mean(np.linalg.norm((objxyz - annot['objCorners3D']), axis=1)))
                print('obj mesh loss:', np.mean(np.linalg.norm((objverts - objMesh.v), axis=1)))

        # ###################### compare GT & prediction on visualization #################
        if(args.figMode !=''):
            fileName = saveFolder + '/' + folder + '_' + file + '.png'
            if(validset == 'evaluation'):
                plot2DforTest(depth, handUVD, objbboxUVD, handmeshUVD, objmeshUVD, fileName)
            else:
                outUVD = {
                    'hand_uvd':handUVD,'handmeshUVD':handmeshUVD,'objbboxUVD':objbboxUVD,'objmeshUVD':objmeshUVD
                }
                gtUVD = {
                    'hand_uvd': gt_hand_uvd, 'handmeshUVD': gt_handMesh_uvd, 'objbboxUVD': gt_obj_uvd, 'objmeshUVD': gt_objmesh_uvd
                }
                if(args.figMode == '2D'):
                    plot2DforValid(depth,gtUVD,outUVD,fileName)
                elif (args.figMode == '3DPC'):
                    plot3DPCforValid(gtUVD,outUVD,fileName)
                elif (args.figMode == '3DMesh'):
                    plot3DMeshforValid(handMesh,objMesh, gtUVD, outUVD, fileName)

        if (args.saveJSON):
            hand_xyz_list.append(handxyz)
            hand_verts_list.append(handverts)
            obj_xyz_list.append(objxyz)
            obj_verts_list.append(objverts)

        #break

    ########## dump results. During testing of qualitative result we don't want to dump ###################
    if(args.saveJSON):
        dump(hand_out_path, hand_xyz_list, hand_verts_list)
        dump(obj_out_path, obj_xyz_list, obj_verts_list)