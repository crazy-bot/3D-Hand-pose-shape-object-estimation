from torch.utils.data import Dataset
import torch
from dataset.HO_Data.data_util import *
import os
import numpy as np
from src.path import OBJ_MODEL_PATH

class Ho3DDataset(Dataset):
    def __init__(self, root, pathFile, augmentation, dtype, isValid=False,preVis=False):
        self.folder = os.path.join(root,'train')
        self.fileNames = os.path.join(root,pathFile)
        self.dtype = dtype
        self.transform = V2VVoxelization(cubic_size=200, augmentation=augmentation)
        self.isValid = isValid
        self.preVis = preVis
        self._load()

    def __getitem__(self, index):
        record = self.filePaths[index]
        # print('record:', record)
        subfolder, file = tuple(record.rstrip().split('/'))
        depthpath = os.path.join(self.folder, subfolder, 'depth', file + '.png')
        annotpath = os.path.join(self.folder, subfolder, 'meta', file + '.pkl')

        depth = read_depth_img(depthpath)
        annot = np.load(annotpath, allow_pickle=True)
        camMat = annot['camMat']
        fx = camMat[0, 0]
        fy = camMat[1, 1]
        ux = camMat[0, 2]
        uy = camMat[1, 2]

        ##################### load object model and annotations #######################
        objMesh = read_obj(
            os.path.join(OBJ_MODEL_PATH, annot['objName'], 'textured_2358.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']

        handJoints = annot['handJoints3D']
        handJoints = handJoints[jointsMapManoToSimple]
        objCorners = annot['objCorners3D']
        _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])

        ################# project given annotations in UVD ###################
        handJoints_uvd = project_3D_points(camMat, handJoints)
        obj_uvd = project_3D_points(camMat, objCorners)
        handMesh_uvd = project_3D_points(camMat, handMesh)
        objmesh_uvd = project_3D_points(camMat, objMesh.v)
        ################ get the common center point of hand and object ###########
        objcenter = np.mean(obj_uvd, axis=0)
        com = np.mean(np.array([handJoints_uvd[0], objcenter]), axis=0)
        # print('com:', com)

        if (not self.isValid):
            ###################### calculate voxel of depthmap and heatmaps of joints and object corners (V2V approach) ############

            ############# project depthmap to 3D world points ###############
            pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
            pointcloud = pointcloud.reshape(-1, 3)
            refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
            refpoint = np.array(refpoint)
            joints_world = Main_pixelToworld(handJoints_uvd.copy(), ux, uy, fx, fy)
            bbox_world = Main_pixelToworld(obj_uvd.copy(), ux, uy, fx, fy)
            handmesh_world = Main_pixelToworld(handMesh_uvd.copy(), ux, uy, fx, fy)
            objmesh_world = Main_pixelToworld(objmesh_uvd.copy(), ux, uy, fx, fy)

            sample = {
                'points': pointcloud,
                'joints': joints_world,
                'bbox': bbox_world,
                'handmesh': handmesh_world,
                'objmesh':objmesh_world,
                'refpoint': refpoint,
            }
            depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel,norm_handmesh,norm_objmesh = self.transform.train_transform(sample)
            dm_voxel = np.zeros(depth_voxel.shape)
            dm_voxel[depth_voxel == 1] = 1
            dm_voxel[mesh_voxel == 1] = 1
            ################ for testing purpose in visualization ###############
            if (self.preVis):
                self.testVis(dm_voxel,heatmap_joints,heatmap_bbox,norm_handmesh,norm_objmesh)

            dm_voxel = torch.from_numpy(dm_voxel.reshape((1, *dm_voxel.shape))).to(self.dtype)
            heatmap_joints = torch.from_numpy(heatmap_joints).to(self.dtype)
            heatmap_bbox = torch.from_numpy(heatmap_bbox).to(self.dtype)
            norm_handmesh = torch.from_numpy(norm_handmesh).to(self.dtype)
            norm_objmesh = torch.from_numpy(norm_objmesh).to(self.dtype)

            return (dm_voxel, heatmap_joints, heatmap_bbox, norm_handmesh,norm_objmesh )
        else:
            pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
            pointcloud = pointcloud.reshape(-1, 3)
            refpoint = Main_pixelToworld(com.reshape(1, -1), ux, uy, fx, fy)
            refpoint = np.array(refpoint)
            joints_world = Main_pixelToworld(handJoints_uvd.copy(), ux, uy, fx, fy)
            bbox_world = Main_pixelToworld(obj_uvd.copy(), ux, uy, fx, fy)
            handmesh_world = Main_pixelToworld(handMesh_uvd.copy(), ux, uy, fx, fy)
            objmesh_world = Main_pixelToworld(objmesh_uvd.copy(), ux, uy, fx, fy)
            sample = {
                'points': pointcloud,
                'joints': joints_world,
                'bbox': bbox_world,
                'handmesh': handmesh_world,
                'objmesh': objmesh_world,
                'refpoint': refpoint
            }
            depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel = self.transform.val_transform(sample)
            dm_voxel = np.zeros(depth_voxel.shape)
            dm_voxel[depth_voxel == 1] = 1
            dm_voxel[mesh_voxel == 1] = 1

            dm_voxel = torch.from_numpy(dm_voxel.reshape((1, *dm_voxel.shape))).to(self.dtype)
            heatmap_joints = torch.from_numpy(heatmap_joints).to(self.dtype)
            heatmap_bbox = torch.from_numpy(heatmap_bbox).to(self.dtype)

            GT = {
                'handverts': handMesh.r.astype(np.float64),
                'objverts': objMesh.v.astype(np.float64),
                'camMat' :annot['camMat'].astype(np.float64),
                'refpoint': refpoint.astype(np.float64)
            }
            return (dm_voxel, heatmap_joints, heatmap_bbox, GT)

    def __len__(self):
        return len(self.filePaths)

    def _load(self):
        self.filePaths = []
        with open(self.fileNames) as f:
            for record in f:
                self.filePaths.append(record)

    def testVis(self, depth_voxel, heatmap_joints, heatmap_bbox, norm_handmesh,norm_objmesh):
        import matplotlib.pyplot as plt
        joints = self.transform.extract_coord_from_output(heatmap_joints)
        objCorners = self.transform.extract_coord_from_output(heatmap_bbox)

        coord_x = np.argwhere(depth_voxel)[:, 0]
        coord_y = np.argwhere(depth_voxel)[:, 1]
        coord_z = np.argwhere(depth_voxel)[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.scatter(coord_x, coord_y, coord_z, c='r', s=10)

        #ax.scatter(norm_handmesh[:,0], norm_handmesh[:,1], norm_handmesh[:,2], c='b', s=10)
        #ax.scatter(norm_objmesh[:, 0], norm_objmesh[:, 1], norm_objmesh[:, 2], c='b', s=10)

        ############## bone joints indexe as pair of points order by thumb, index, middle, ring, pinky fingers
        bones_3d = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        edges_3d = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        ############ Display in 3D

        for b in bones_3d:
            ax.plot(joints[b, 0], joints[b, 1], joints[b, 2], linewidth=1.0, c='g')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="b", edgecolors='b')

        for e in edges_3d:
            ax.plot(objCorners[e, 0], objCorners[e, 1], objCorners[e, 2], linewidth=1.0, c='m')
        ax.scatter(objCorners[:, 0], objCorners[:, 1], objCorners[:, 2], c="c", edgecolors='c')

        plt.show()

class V2VVoxelization(object):
    def __init__(self, cubic_size, augmentation=False):
        self.cubic_size = cubic_size
        self.cropped_size, self.original_size = 44, 48
        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
        self.pool_factor = 1
        self.std = 1.7
        self.augmentation = augmentation
        self.extract_coord_from_output = extract_coord_from_output
        output_size = int(self.cropped_size / self.pool_factor)
        # Note, range(size) and indexing = 'ij'
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size),
                                     indexing='ij')

    def train_transform(self, sample):
        points, joints, bbox, handmesh,objmesh, refpoint = sample['points'], sample['joints'], \
                                                           sample['bbox'], sample['handmesh'],sample['objmesh'], sample['refpoint']

        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size / 2 - self.cropped_size / 2
        else:
            ## Augmentations
            # Resize
            new_size = np.random.rand() * 40 + 80

            # Rotation
            angle = np.random.rand() * 80 / 180 * np.pi - 40 / 180 * np.pi

            # Translation
            trans = np.random.rand(3) * (self.original_size - self.cropped_size)

        depth_voxel = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        heatmap_joints = generate_heatmap_gt(joints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
                                             self.pool_factor, self.std)
        heatmap_bbox = generate_heatmap_gt(bbox, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
                                           self.pool_factor, self.std)
        fullmesh = np.concatenate([handmesh,objmesh],axis=0)
        mesh_voxel = generate_cubic_input(fullmesh, refpoint, new_size, angle, trans, self.sizes)
        norm_handmesh = generate_coord(handmesh,refpoint,new_size, angle, trans, self.sizes)
        norm_objmesh = generate_coord(objmesh, refpoint, new_size, angle, trans, self.sizes)

        return depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel, norm_handmesh,norm_objmesh

    def val_transform(self, sample):
        points, joints, bbox, handmesh,objmesh, refpoint = sample['points'], sample['joints'], \
                                                           sample['bbox'], sample['handmesh'],sample['objmesh'], sample['refpoint']

        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size / 2 - self.cropped_size / 2
        else:
            ## Augmentations
            # Resize
            new_size = np.random.rand() * 40 + 80

            # Rotation
            angle = np.random.rand() * 80 / 180 * np.pi - 40 / 180 * np.pi

            # Translation
            trans = np.random.rand(3) * (self.original_size - self.cropped_size)

        depth_voxel = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        heatmap_joints = generate_heatmap_gt(joints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
                                             self.pool_factor, self.std)
        heatmap_bbox = generate_heatmap_gt(bbox, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
                                           self.pool_factor, self.std)
        fullmesh = np.concatenate([handmesh,objmesh],axis=0)
        mesh_voxel = generate_cubic_input(fullmesh, refpoint, new_size, angle, trans, self.sizes)

        return depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel
