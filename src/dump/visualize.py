from util.main_util import *
from util.contactloss import compute_contact_loss
from util.voxel_util import *
from dataset.HO_Data.data_util import *
import torch
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    mode = 'train'
    train_folder = '/data/Guha/HO3D_v2/{}/'.format(mode)
    train_txt =  '/data/Guha/HO3D_v2/{}.txt'.format('valid')
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

    voxelization_train = V2VVoxelization(cubic_size=200, augmentation=False)
    with open(train_txt) as tf:
        for record in tf:
            print(record)
            folder,file = tuple(record.rstrip().split('/'))
            depthpath = os.path.join(train_folder,folder,'depth',file+'.png')
            annotpath = os.path.join(train_folder,folder,'meta',file+'.pkl')
            depth = read_depth_img(depthpath)
            annot = np.load(annotpath, allow_pickle=True)
            camMat = annot['camMat']
            fx = camMat[0, 0]
            fy = camMat[1, 1]
            ux = camMat[0, 2]
            uy = camMat[1, 2]
            if(mode != 'eval'):
                handJoints = annot['handJoints3D']
                handJoints = handJoints[jointsMapManoToSimple]
                objCorners = annot['objCorners3D']
                objname = annot['objName']
                #handMesh = annot['handVerts']
                _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])
                print('handmesh ', handMesh.shape)

                # objMesh = []
                # with open('/data/Guha/YCB_Video_Models/models/{}/points.xyz'.format(objname)) as objfile:
                #     for line in objfile:
                #         temp = line.strip().split()
                #         objMesh.append(temp)
                # objMesh = np.array(objMesh).astype(float)
                # objMesh = np.matmul(objMesh, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']
                # objmesh_uvd = project_3D_points(camMat, objMesh)
                ##################### load object model
                objMesh = read_obj(os.path.join('/data/Guha/YCB_Video_Models/models', annot['objName'], 'textured_2358.obj'))
                # rot = R.from_rotvec(annot['objRot'].T)
                # quat = rot.as_quat()
                # #trans3D = annot['objTrans'].reshape(1,-1).dot(coord_change_mat.T)
                # objtrans_uvd =  project_3D_points(camMat, annot['objTrans'].reshape(1,-1))
                objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T)+annot['objTrans']
                objmesh_uvd = project_3D_points(camMat, objMesh.v)

                handJoints_uvd = project_3D_points(camMat, handJoints)
                handMesh_uvd = project_3D_points(camMat,handMesh)
                obj_uvd = project_3D_points(camMat, objCorners)
                objcenter = np.mean(obj_uvd, axis=0)
                com = np.mean(np.array([handJoints_uvd[0], objcenter]), axis=0)
                #com = handJoints_uvd[0]
                #com = np.mean(handJoints_uvd, axis=0)
                print ('com:', com)

                #################### compute contact loss #################
                a = torch.from_numpy(handMesh_uvd.astype(np.float32)).unsqueeze(0)
                b = torch.from_numpy(handMesh.f.astype(int))
                c = torch.from_numpy(objmesh_uvd.astype(np.float32)).unsqueeze(0)
                d = torch.from_numpy(objMesh.f.astype(int))
                compute_contact_loss(a,b,c,d)

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plot3dVisualize(ax, handMesh, verts=handMesh_uvd, flip_x=False, isOpenGLCoords=False, c="r")
                # plt.show()
                #open3dVisualize([handMesh],['r'])

            else:
                handbox2D= annot['handBoundingBox']
                objCorners = annot['objCorners3D']
                obj_uvd = project_3D_points(camMat, objCorners)
                handroot = np.array(annot['handJoints3D'])
                handroot_uvd = project_3D_points(camMat, handroot.reshape(1,-1))

                # handCenter = np.mean(np.array(handbox2D).reshape(2,2),axis=1)
                # z = depth[int(handCenter[0]), int(handCenter[1])]
                # handCenter = np.hstack((handCenter,z))
                objcenter = np.mean(obj_uvd,axis=0)
                com = np.mean(np.array([handroot_uvd[0], objcenter]), axis=0)

                print('com:', com)

            ################## draw mesh on orginal image ############

            #drawSkeleton(handJoints_uvd,obj_uvd,depth,com)

            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            #plt.imshow(depth, 'gray')
            ax.scatter(objmesh_uvd[:, 0], objmesh_uvd[:, 1],objmesh_uvd[:, 2], c='g')
            #ax.scatter(handMesh_uvd[:, 0], handMesh_uvd[:, 1], handMesh_uvd[:, 2], c='b')
            # ax.scatter(handJoints_uvd[:, 0], handJoints_uvd[:, 1],handJoints_uvd[:, 2], c='r')
            # ax.scatter(obj_uvd[:, 0], obj_uvd[:, 1],obj_uvd[:, 2], c='m')
            plot3dVisualize(ax, handMesh, verts=handMesh_uvd, flip_x=False, isOpenGLCoords=False, c="r")

            plt.show()
            #################### v2v approach : voxel segment ###############
            # pointcloud = depthmap2points(depth, ux,uy, fx, fy)
            # pointcloud = pointcloud.reshape(-1, 3)
            #
            # h, w = depth.shape
            # refpoint = [pixel2world(com[0], com[1], com[2], ux, uy, fx, fy)]
            # refpoint = np.array(refpoint)
            # joints_world = np.zeros((21, 3), dtype=np.float32)
            # joints_world[:, 0], joints_world[ :, 1], joints_world[:, 2] = pixel2world(handJoints_uvd[:,0],handJoints_uvd[:,1],handJoints_uvd[:,2],ux,uy,fx,fy)
            # corners_world = np.zeros((8, 3), dtype=np.float32)
            # corners_world[:, 0], corners_world[:, 1], corners_world[:, 2] = pixel2world(obj_uvd[:, 0],
            #                                                                          obj_uvd[:, 1],
            #                                                                          obj_uvd[:, 2], ux, uy, fx, fy)
            # handMesh_world = np.zeros((778, 3), dtype=np.float32)
            # handMesh_world[:, 0], handMesh_world[:, 1], handMesh_world[:, 2] = pixel2world(handMesh_uvd[:, 0],
            #                                                                             handMesh_uvd[:, 1],
            #                                                                             handMesh_uvd[:, 2], ux, uy, fx, fy)
            ##################### Main_pixelToworld ############################
            pointcloud = Main_depthmap2points(depth, ux, uy, fx, fy)
            pointcloud = pointcloud.reshape(-1, 3)

            h, w = depth.shape
            refpoint = Main_pixelToworld(com.reshape(1,-1), ux, uy, fx, fy)
            refpoint = np.array(refpoint)
            joints_world = np.zeros((21, 3), dtype=np.float32)
            joints_world = Main_pixelToworld(handJoints_uvd.copy(), ux, uy, fx,fy)
            corners_world = np.zeros((8, 3), dtype=np.float32)
            corners_world = Main_pixelToworld(obj_uvd.copy(), ux, uy, fx, fy)
            handMesh_world = np.zeros((778, 3), dtype=np.float32)
            handMesh_world = Main_pixelToworld(handMesh_uvd.copy(), ux, uy, fx, fy)
            objMesh_world = np.zeros((778, 3), dtype=np.float32)
            objMesh_world = Main_pixelToworld(objmesh_uvd.copy(), ux, uy, fx, fy)
            objtrans_world = Main_pixelToworld(objtrans_uvd.copy(), ux, uy, fx, fy)
            #objtrans_uvd2 = Main_worldTopixel(objtrans_world,ux, uy, fx, fy)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            #plt.imshow(depth, 'gray')
            ax.scatter( objtrans_uvd[:,0],  objtrans_uvd[:,1],objtrans_uvd[:,2], c='g',s=20)
            #ax.scatter( annot['objTrans'][ 0],  annot['objTrans'][1],  annot['objTrans'][ 2], c='g', s=20)
            ax.scatter(objmesh_uvd[:, 0], objmesh_uvd[:, 1],objmesh_uvd[:, 2], c='b',alpha=0.01)
            # plt.scatter(joints_world[:, 0], joints_world[:, 1], c='r')
            # plt.scatter(corners_world[:, 0], corners_world[:, 1], c='m')
            plt.show()


            sample = {
                'points': pointcloud,
                'joints': joints_world,
                # 'corners':corners_world,
                'handMesh': handMesh_world,
                'refpoint': refpoint,
                'handJoints_uvd':handJoints_uvd,
                'handMesh_uvd': handMesh_uvd,
                'refpoint_uvd': com,
            }
            voxel,heatmap_joints,norm_handMesh,voxel44,voxel_mesh,manoPose,manoShape = voxelization_train(sample)
            manoPose = manoPose[jointsMapManoToSimple]
            #voxel, heatmapjoints,heatmap2joints = voxelization_train(sample)


            # jointsuvd2 = points2pixels(jointsworld2,w,h,fx,fy)

            ######### voxel as pointcloud #########
            #voxel = np.resize(voxel,(44,44,44))
            heatmap2joints = extract_coord_from_output(heatmap_joints)*2
            norm_handMesh = norm_handMesh *2
            coord_x = np.argwhere(voxel)[:,0]
            coord_y = np.argwhere(voxel)[:,1]
            coord_z = np.argwhere(voxel)[:,2]
            print(len(coord_x))
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.scatter(coord_x,coord_y,coord_z,c='r',s=10)
            # coord_x = np.argwhere(voxel_mesh)[:, 0]
            # coord_y = np.argwhere(voxel_mesh)[:, 1]
            # coord_z = np.argwhere(voxel_mesh)[:, 2]
            # ax.scatter(coord_x, coord_y, coord_z, c='b', s=10)
            #ax.voxels(voxel,facecolors='r',  edgecolor='b')
            #ax.scatter(voxel[:,0], voxel[:,1],voxel[:,2], c='r')
            ax.scatter(heatmap2joints[:,0],heatmap2joints[:,1],heatmap2joints[:,2], c='b', s= 30)
            ax.scatter(norm_handMesh[:,0],norm_handMesh[:,1],norm_handMesh[:,2], c='b',s=10)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            # coord_x = np.argwhere(voxel)[:, 0]
            # coord_y = np.argwhere(voxel)[:, 1]
            # coord_z = np.argwhere(voxel)[:, 2]
            # ax.scatter(coord_x, coord_y, coord_z, c='b', s=10)
            ax.scatter(handJoints_uvd[:,0],handJoints_uvd[:,1],handJoints_uvd[:,2], c='b',s=30)
            ax.scatter(handMesh_uvd[:,0],handMesh_uvd[:,1],handMesh_uvd[:,2], c='b', s= 10)


            angle = voxelization_train.angle
            theta ,trans, beta = annot['handPose'], annot['handTrans'], annot['handBeta']
            theta = theta.reshape(16,3)
            org_keypoints = theta.copy()
            theta[:, 0] = org_keypoints[:, 0] * np.cos(angle) - org_keypoints[:, 1] * np.sin(angle)
            theta[:, 1] = org_keypoints[:, 0] * np.sin(angle) + org_keypoints[:, 1] * np.cos(angle)
            joints, handMesh = forwardKinematics(theta.reshape(-1),trans,beta)
            joints = project_3D_points(camMat, joints)
            handMesh = project_3D_points(camMat, handMesh)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='b', s=30)
            ax.scatter(handMesh[:, 0], handMesh[:, 1], handMesh[:, 2], c='b', s=10)

            plt.show()







