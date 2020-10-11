from __future__ import print_function, unicode_literals
from config import *
import sys
sys.path.append(PROJECT_PATH)

import argparse
from tqdm import tqdm
from dataset.HO_Data.codelab_util import *
from dataset.HO_Data.convert import *
from networks.HO_Nets.HO_ShapeNet import HO_ShapeNet
from networks.HO_Nets.HO_Posenet import HO_Posenet
from src.config import *
import torch
import json

def loadModel(device, dtype):
    print('==> Constructing model ..HO_Posenet')
    posenet = HO_Posenet(input_channels=1, hand_channels=handpoints_num, obj_channels=objpoints_num)
    posenet = torch.nn.DataParallel(posenet)

    print('==> Constructing model ..HO_ShapeNet_v1')
    shapenet = HO_ShapeNet(input_channels=1, hand_channels=no_handverts , obj_channels=no_objverts )
    shapenet = torch.nn.DataParallel(shapenet)

    posenet = posenet.to(device, dtype)
    shapenet = shapenet.to(device, dtype)

    ################ load predefined checkpoint ###############
    posenet.load_state_dict(torch.load(PNet_ckpt)['model_state_dict'])
    shapenet.load_state_dict(torch.load(SNet_v1_ckpt)['model_state_dict'])

    return posenet, shapenet


def main(args):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    set_name = 'evaluation'
    # init output containers
    hand_xyz_list, hand_verts_list = list(), list()
    obj_xyz_list, obj_verts_list = list(), list()

    # read list of evaluation files
    with open(os.path.join(DATA_DIR, set_name + '.txt')) as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]

    assert len(file_list) == db_size(set_name), '%s.txt is not accurate. Aborting' % set_name

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float

    transform = V2VVoxelization(cubic_size=200, augmentation=False)
    posenet, shapenet = loadModel(device, dtype)

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        if idx >= db_size(set_name):
            break

        seq_name = file_list[idx].split('/')[0]
        file_id = file_list[idx].split('/')[1]
        print(seq_name, file_id)

        # load input image
        depthpath = os.path.join(DATA_DIR, set_name, seq_name, 'depth', file_id + '.png')
        annotpath = os.path.join(DATA_DIR, set_name, seq_name, 'meta', file_id + '.pkl')
        depth = read_depth_img(depthpath)
        annot = np.load(annotpath, allow_pickle=True)
        camMat = annot['camMat']
        fx = camMat[0, 0]
        fy = camMat[1, 1]
        ux = camMat[0, 2]
        uy = camMat[1, 2]

        objCorners = annot['objCorners3D']
        obj_uvd = project_3D_points(camMat, objCorners)
        handroot = np.array(annot['handJoints3D'])
        handroot_uvd = project_3D_points(camMat, handroot.reshape(1, -1))
        handroot_uvd = handroot_uvd[0]
        objcenter = np.mean(obj_uvd, axis=0)
        com = np.mean(np.array([handroot_uvd, objcenter]), axis=0)

        # print('objCorners', objCorners)
        # print('handroot', handroot)
        ###################### calculate voxel of depthmap (V2V approach) ############
        refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
        refpoint = np.array(refpoint)
        pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
        pointcloud = pointcloud.reshape(-1, 3)

        voxel88 = transform.voxelize88(pointcloud, refpoint)
        voxel88 = torch.Tensor(voxel88).unsqueeze(0).to(device, dtype)

        voxel44 = transform.voxelize44(pointcloud, refpoint)
        voxel44 = torch.Tensor(voxel44).unsqueeze(0).to(device, dtype)

        ################# raw prediction of joint heatmaps and hand mesh ###################
        with torch.no_grad():
            poseResult = posenet(voxel88)
            shapeResult = shapenet(voxel44)

        #################### convert into submission format #####################
        ###################### hand conversion ####################
        hand_hmp = poseResult['handpose'][0].cpu().numpy()
        hand_keypoints_uvd, handxyz = pred2Org_handjoints(hand_hmp, refpoint, ux, uy, fx, fy)

        handmesh = shapeResult['handverts'][0].cpu().numpy()
        handmeshUVD, handverts = pred2Org_mesh(handmesh, refpoint, ux, uy, fx, fy)

        ################# object conversion #######################
        obj_hmp = poseResult['objpose'][0].cpu().numpy()
        objbboxUVD, objxyz = pred2Org_objbbox(obj_hmp, refpoint, ux, uy, fx, fy)

        objmesh = shapeResult['objverts'][0].cpu().numpy()
        objmeshUVD, objverts = pred2Org_mesh(objmesh, refpoint, ux, uy, fx, fy)

        ######################  prediction on visualization #################
        if(args.saveFig !=''):
            fileName = args.saveFig+ '/' +seq_name + '_' + file_id + '.png'
            plot2DforTest(depth, hand_keypoints_uvd,objbboxUVD,handmeshUVD,objmeshUVD,fileName)

        ################# simple check if xyz and verts are in opengl coordinate system
        # print('objxyz', objxyz)
        # print('handxyz', handxyz[0])
        if np.all(handxyz[:, 2] > 0) or np.all(handverts[:, 2] > 0) or np.all(objxyz[:, 2] > 0) or np.all(
                objverts[:, 2] > 0):
            raise Exception(
                'It appears the pose estimates are not in OpenGL coordinate system. Please read README.txt in dataset folder. Aborting!')

        hand_xyz_list.append(handxyz)
        hand_verts_list.append(handverts)
        obj_xyz_list.append(objxyz)
        obj_verts_list.append(objverts)
        #break

    ########## dump results
    hand_out_path = PROJECT_PATH+'/'+args.handout
    obj_out_path = PROJECT_PATH + '/' + args.objout
    dump(hand_out_path, hand_xyz_list, hand_verts_list)
    dump(obj_out_path, obj_xyz_list, obj_verts_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def pred_template(img, aux_info):
    """ Predict joints and vertices from a given sample.
        img: (640, 480, 3) RGB image.
        aux_info: dictionary containing hand bounding box, camera matrix and root joint 3D location
    """
    # TODO: Put your algorithm here, which computes (metric) 3D joint coordinates and 3D vertex positions
    xyz = np.zeros((21, 3))  # 3D coordinates of the 21 joints
    verts = np.zeros((778, 3))  # 3D coordinates of the shape vertices
    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--handout', type=str, default='hand_baseline.json', help='File to save the predictions.')
    parser.add_argument('--objout', type=str, default='obj_baseline.json', help='File to save the predictions.')
    parser.add_argument('--saveFig', type=str, default='', help='Folder to save the images.')
    args = parser.parse_args()

    # call with a predictor function
    main(args)