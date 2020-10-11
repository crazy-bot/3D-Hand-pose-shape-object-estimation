from config import *
import sys
sys.path.append(PROJECT_PATH)

import torch
from dataset.HO_Data.codelab_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_Posenet import HO_Posenet
from networks.HO_Nets.HO_VoxelNet import HO_VoxelNet
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet
from networks.HO_Nets.HO_PSNet import HO_PSNet
from networks.HO_Nets.HO_PVNet import HO_PVNet
import argparse
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('--saveJSON', type=bool, default=False, help='save json or not')
parser.add_argument('--figMode', default='2D',choices=['2D', '3DPC', '3DMesh',''])
parser.add_argument('--handout', type=str, default='pip_hand_snetv6_{}.json'.format(validset), help='File to save the predictions.')
parser.add_argument('--objout', type=str, default='pip_obj_snetv6_{}.json'.format(validset), help='File to save the predictions.')

args = parser.parse_args()
#######################################################################################
###################### Model weights load ###########################

# hops_weights = torch.load(PSNet_ckpt)['model_state_dict']
# hops_weights = {k.replace('module.',''):hops_weights[k] for k in list(hops_weights.keys()) }
#posenet_weights = {k.replace('poseNet.',''):hops_weights[k] for k in list(hops_weights.keys()) if 'poseNet' in k }

# hopv_weights = torch.load(PVNet_ckpt)['model_state_dict']
# hopv_weights = {k.replace('module.',''):hopv_weights[k] for k in list(hopv_weights.keys()) }
# voxelnet_weights = {k.replace('voxelNet.',''):hopv_weights[k] for k in list(hopv_weights.keys()) if 'voxelNet' in k }

# posenet_weights = torch.load(PNet_ckpt)['model_state_dict']
# posenet_weights = {k.replace('module.',''):posenet_weights[k] for k in list(posenet_weights.keys()) }
#
# voxelnet_weights = torch.load(VNet_ckpt)['model_state_dict']
# voxelnet_weights = {k.replace('module.',''):voxelnet_weights[k] for k in list(voxelnet_weights.keys()) }

shapenet_weights = torch.load(SNet_v6_ckpt)['model_state_dict']
shapenet_weights = {k.replace('module.',''):shapenet_weights[k] for k in list(shapenet_weights.keys()) }

pvnet_weights = torch.load(PVNet_ckpt)['model_state_dict']
pvnet_weights = {k.replace('module.',''):pvnet_weights[k] for k in list(pvnet_weights.keys()) }


# print('==> Constructing model ..HO_Posenet')
# posenet = HO_Posenet(input_channels=1, hand_channels=handpoints_num, obj_channels=objpoints_num)
# print('==> Constructing model ..HO_VoxelNet')
# voxelnet= HO_VoxelNet(input_channels=1+21+8)
print('==> Constructing model ..HO_ShapeNet_v3')
shapenet = HO_ShapeNet(input_channels=1+21+8, hand_channels=no_handverts, obj_channels=no_objverts)
print('==> Constructing model ..HO_PVNet')
pvnet = HO_PVNet()

#posenet = posenet.to(device, dtype)
shapenet = shapenet.to(device, dtype)
#voxelnet = voxelnet.to(device,dtype)
pvnet = pvnet.to(device, dtype)
# posenet.load_state_dict(posenet_weights)
# voxelnet.load_state_dict(voxelnet_weights)
shapenet.load_state_dict(shapenet_weights)
pvnet.load_state_dict(pvnet_weights)
#######################################################################################
#####################  Validate #################
print('==> Testing ..')
folder_path = DATA_DIR + '/' + validset
file_path = DATA_DIR + '/' + validset + '.txt'
transform = V2VVoxelization(cubic_size=200, augmentation=False)
meanHandloss,meanObjloss = [],[]

# init output containers
hand_xyz_list, hand_verts_list = list(), list()
obj_xyz_list, obj_verts_list = list(), list()
hand_out_path = PROJECT_PATH+'/'+args.handout
obj_out_path = PROJECT_PATH + '/' + args.objout

############### savefig dir ###########
if (args.figMode != ''):
    saveFolder = SAVEFIG_DIR + '/pipSNet_v6/'
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
            # poseresult = posenet(voxel88)
            # in2 = torch.cat((voxel44, poseresult['handpose'], poseresult['objpose']), dim=1)
            # out_voxel = voxelnet(in2)
            pvResult = pvnet(voxel88, voxel44)
            out_voxel = pvResult['voxel']
            ######### voxel enrichment ##########
            #out_voxel = torch.nn.Sigmoid()(out_voxel)
            # out_voxel[out_voxel>0.9] = 1
            # out_voxel[out_voxel <= 0.9] = 0
            voxel44[out_voxel >0.8 ] = 1



            # psResult = psnet(voxel88, voxel44)
            # out_handmesh = psResult['handverts'][0].cpu().numpy()
            # out_objmesh = psResult['objverts'][0].cpu().numpy()
            # out_fullmesh = np.concatenate([out_handmesh,out_objmesh])
            # cubic = scattering(out_fullmesh, 44)
            # mesh_voxel = torch.from_numpy(cubic.reshape((1, 1, *cubic.shape))).to(device,dtype)
            # voxel44[mesh_voxel==1] = 1
            ######vis
            # joints = extract_coord_from_output(psResult['handpose'][0].cpu().numpy())
            # objCorners = extract_coord_from_output(psResult['objpose'][0].cpu().numpy())
            # testVis2(voxel44.squeeze().cpu().numpy(), joints, objCorners)


            in3 = torch.cat((voxel44, pvResult['handpose'], pvResult['objpose']), dim=1)
            shaperesult = shapenet(in3)

            ##################### post processing of outputs ################
            ###################### hand conversion ####################
            hand_hmp = pvResult['handpose'][0].cpu().numpy()
            handUVD, handxyz = pred2Org_handjoints(hand_hmp, refpoint, ux, uy, fx, fy)

            handmesh = shaperesult['handverts'][0].cpu().numpy()
            handmeshUVD, handverts = pred2Org_mesh(handmesh, refpoint, ux, uy, fx, fy)

            ################# object conversion #######################
            obj_hmp = pvResult['objpose'][0].cpu().numpy()
            objbboxUVD, objxyz = pred2Org_objbbox(obj_hmp, refpoint, ux, uy, fx, fy)

            objmesh = shaperesult['objverts'][0].cpu().numpy()
            objmeshUVD, objverts = pred2Org_mesh(objmesh, refpoint, ux, uy, fx, fy)

            if (validset != 'evaluation'):
                print('hand points loss:', np.mean(np.linalg.norm((handxyz - annot['handJoints3D']), axis=1)))
                print('hand mesh loss:', np.mean(np.linalg.norm((handverts - handMesh), axis=1)))
                print('obj points loss:', np.mean(np.linalg.norm((objxyz - annot['objCorners3D']), axis=1)))
                print('obj mesh loss:', np.mean(np.linalg.norm((objverts - objMesh.v), axis=1)))

            ######################### compare GT & prediction on visualization #################
            if (args.figMode != ''):
                fileName = saveFolder + '/' + folder + '_' + file + '.png'
                if (validset == 'evaluation'):
                    plot2DforTest(depth, handUVD, objbboxUVD, handmeshUVD, objmeshUVD, fileName)
                else:
                    outUVD = {
                        'hand_uvd': handUVD, 'handmeshUVD': handmeshUVD, 'objbboxUVD': objbboxUVD,
                        'objmeshUVD': objmeshUVD
                    }
                    gtUVD = {
                        'hand_uvd': gt_hand_uvd, 'handmeshUVD': gt_handMesh_uvd, 'objbboxUVD': gt_obj_uvd,
                        'objmeshUVD': gt_objmesh_uvd
                    }
                    if (args.figMode == '2D'):
                        plot2DforValid(depth, gtUVD, outUVD, fileName)
                    elif (args.figMode == '3DPC'):
                        plot3DPCforValid(gtUVD, outUVD, fileName)
                    elif (args.figMode == '3DMesh'):
                        plot3DMeshforValid(handMesh, objMesh, gtUVD, outUVD, fileName)

        if (args.saveJSON):
            hand_xyz_list.append(handxyz)
            hand_verts_list.append(handverts)
            obj_xyz_list.append(objxyz)
            obj_verts_list.append(objverts)

            # break

        ########## dump results. During testing of qualitative result we don't want to dump ###################
    if (args.saveJSON):
        dump(hand_out_path, hand_xyz_list, hand_verts_list)
        dump(obj_out_path, obj_xyz_list, obj_verts_list)