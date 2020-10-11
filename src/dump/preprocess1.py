import os
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fx = 614.62700
fy = 614.10100
img_size = 150
cube_size = 140
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2] , proj_pts[:,2]*1000],axis=1)

    #new_proj = projectPoints(pts3D,cam_mat)
    assert len(proj_pts.shape) == 2
    return proj_pts

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def world2pixel(x, fx, fy, ux, uy):
    x[ :, 0] = (x[ :, 0] * fx / x[ :, 2]) + ux
    x[ :, 1] = (x[ :, 1] * fy / x[ :, 2]) + uy
    #return x.astype(int)
    return x

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""
    #depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    #_assert_exist(depth_filename)
    print (depth_filename)
    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256

    dpt = dpt * depth_scale * 1000

    dpt[dpt==0] = 2000
    return dpt


def CropImage(image,com):

  u , v, d = com
  u = float(u)
  v = float(v)
  d = float(d)

  xstart = u - float(cube_size) / d * fx
  xend = u + float(cube_size) / d * fx
  ystart = v - float(cube_size) / d * fy
  yend = v + float(cube_size) / d * fy
  print ('xstart,xend,ystart,yend',xstart,xend,ystart,yend)

  src = [(xstart, ystart), (xstart, yend), (xend, ystart)]
  dst = [(0, 0), (0, img_size - 1), (img_size - 1, 0)]
  trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
          np.array(dst, dtype=np.float32))
  res_img = cv2.warpAffine(image, trans, (img_size, img_size), None,
          cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, d + cube_size)
  res_img -= d
  res_img = np.maximum(res_img, -cube_size)
  res_img = np.minimum(res_img, cube_size)
  res_img /= cube_size

  return res_img,trans

##################### normalizing hand annotations as per cropimage ################
def normalizeAnnot(M,com,joints,corners):
    print ('original joints: ',joints)
    u, v, d = com
    d = float(d)

    depth_hand = joints[:, 2].copy()
    depth_hand -= d
    depth_hand /= cube_size

    joints[:, 2] = 1
    trans_jonts = np.dot(M, joints.T)
    trans_jonts /= cube_size
    trans_jonts = trans_jonts.T
    joints = np.hstack((trans_jonts, depth_hand.reshape(-1,1)))

    depth_obj = corners[:, 2].copy()
    depth_obj -= d
    depth_obj /= cube_size

    corners[:, 2] = 1
    trans_corners = np.dot(M, corners.T)
    trans_corners /= cube_size
    trans_corners = trans_corners.T
    corners = np.hstack((trans_corners, depth_obj.reshape(-1,1)))
    print('normalized joints',joints)
    return joints,corners

 ##################### de-normalizing hand annotations as per cropimage ################
def deNormalizeAnnot(M, com, joints, corners):
    u, v, d = com
    d = float(d)
    invM = cv2.invertAffineTransform(M)

    depth_hand = joints[:, 2].copy()
    depth_hand *= cube_size
    depth_hand += d

    joints *= cube_size
    joints[:, 2] = 1
    trans_jonts = np.dot(invM, joints.T)
    trans_jonts = trans_jonts.T
    joints = np.hstack((trans_jonts, depth_hand.reshape(-1,1)))

    depth_obj = corners[:, 2].copy()
    depth_obj *= cube_size
    depth_obj += d

    corners *= cube_size
    corners[:, 2] = 1
    trans_corners = np.dot(invM, corners.T)
    trans_corners = trans_corners.T
    corners = np.hstack((trans_corners, depth_obj.reshape(-1,1)))
    print ('denormalized joints: ',joints)
    return joints, corners

def getPalmCenter(joints):
    joint_idx = [0,2,6,10,14,18]
    mean = []
    for i in range(1,6):
        m = np.mean([joints[0],joints[joint_idx[i]]],axis=0)
        mean.append(m)
    mean = np.asarray(mean)
    return np.mean(mean,axis=0)

def getObjCenter(corners):
    corners = np.asarray(corners)
    return np.mean(corners,axis=0)

def getCenter(joints,corners):
    palmcenter = getPalmCenter(joints)
    objcenter = getObjCenter(corners)
    return np.mean(np.array([palmcenter,objcenter]),axis=0)

#################### draw skeleton and object bounding box
def drawSkeleton(joints,objCorners,img, com, mode='2D'):
    fig = plt.figure()
    palmcenter = getPalmCenter(joints)
    objcenter = getObjCenter(objCorners)
    # print('palmcenter ',palmcenter)
    # print('objcenter ', objcenter)
    #joints = np.append(joints, [palmcenter], axis=0)

    if(mode == '2D'):
        # bone joints indexe as set of points order by thumb, index, middle, ring, pinky fingers
        bones = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],[0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
        jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        # skeleton array to hold the x,y coordinates
        ske_joint = []
        obj_edges = []
        ####### add hand bones
        for i in range(0, len(bones)):
            pairset = bones[i]
            X = []
            Y = []
            for j in pairset:
                X.append(joints[j][0])
                Y.append(joints[j][1])
            ske_joint.append([X, Y])

        ########### add object corners
        for i in range(0, len(jointConns)):
            pairset = jointConns[i]
            X = []
            Y = []
            for j in pairset:
                X.append(objCorners[j][0])
                Y.append(objCorners[j][1])
            obj_edges.append([X, Y])

        ############# scatter the points
        fig.suptitle('original', fontsize=14, fontweight='bold')
        plt.imshow(img,'gray')
        plt.scatter(joints[:, 0], joints[:, 1], s=30, c='y',edgecolors='y', alpha=0.5)
        plt.scatter(palmcenter[0], palmcenter[1], s=30, c='r')

        plt.scatter(objCorners[:,0],objCorners[:,1],s=30, c='c')
        plt.scatter(objcenter[0], objcenter[1], s=30, c='r')

        plt.scatter(com[0], com[1], s=30, c='b')

        ############ draw the bones
        for s in ske_joint:
            plt.plot(s[0], s[1], linewidth=1.0,c='g')
        for o in obj_edges:
            plt.plot(o[0], o[1], linewidth=1.0, c='m')
    else:

        ############## bone joints indexe as pair of points order by thumb, index, middle, ring, pinky fingers
        bones_3d = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        edges_3d = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]

        ############ Display in 3D
        ax = fig.add_subplot((111), projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for b in bones_3d:
            ax.plot(joints[b, 0], joints[b, 1], joints[b, 2], linewidth=1.0 , c='g')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],c="b", edgecolors='b')

        for e in edges_3d:
            ax.plot(objCorners[e, 0], objCorners[e, 1], objCorners[e, 2], linewidth=1.0 , c='m')
        ax.scatter(objCorners[:, 0], objCorners[:, 1], objCorners[:, 2],c="c", edgecolors='c')

        ax.scatter(palmcenter[0],palmcenter[1],palmcenter[2],c='r')
        ax.scatter(objcenter[0], objCorners[1], palmcenter[2], c='r')
        ax.scatter(com[0], com[1], com[2], c='b')

    plt.ion()
    plt.draw()
    plt.pause(1)
    plt.waitforbuttonpress()
    plt.close('all')
    #return plt

def drawProcessedImg(joints,objCorners,img,mode='2D' ):
    fig = plt.figure()
    img_scaled = (img + 1) / 2
    im = np.zeros((img.shape[0], img.shape[1], 3))
    im[:, :, 0] = img_scaled
    im[:, :, 1] = img_scaled
    im[:, :, 2] = img_scaled

    copy_joints = joints.copy()
    copy_objCorners = objCorners.copy()
    copy_joints[:, 0] = copy_joints[:, 0]  * img.shape[0]
    copy_joints[:, 1] = copy_joints[:, 1] * img.shape[1]
    copy_objCorners[:, 0] = copy_objCorners[:, 0] * img.shape[0]
    copy_objCorners[:, 1] = copy_objCorners[:, 1] * img.shape[1]

    if (mode == '2D'):
        # bone joints indexe as set of points order by thumb, index, middle, ring, pinky fingers
        bones = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
        jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        # skeleton array to hold the x,y coordinates
        ske_joint = []
        obj_edges = []
        for i in range(0, len(bones)):
            pairset = bones[i]
            X = []
            Y = []
            for j in pairset:
                X.append(copy_joints[j][0])
                Y.append(copy_joints[j][1])
                ske_joint.append([X, Y])

        ########### add object corners
        for i in range(0, len(jointConns)):
            pairset = jointConns[i]
            X = []
            Y = []
            for j in pairset:
                X.append(copy_objCorners[j][0])
                Y.append(copy_objCorners[j][1])
            obj_edges.append([X, Y])

        ############# scatter the points
        fig.suptitle('original', fontsize=14, fontweight='bold')
        plt.clf()
        plt.imshow(img_scaled,'gray')
        plt.scatter(copy_joints[:, 0], copy_joints[:, 1], s=30, c='y',edgecolors='y', alpha=0.5)
        plt.scatter(copy_objCorners[:,0],copy_objCorners[:,1],s=30, c='c')

        ############ draw the bones
        for s in ske_joint:
            plt.plot(s[0], s[1], linewidth=1.0, c='g')
        for o in obj_edges:
            plt.plot(o[0], o[1], linewidth=1.0, c='m')
    else:

        ############## bone joints indexe as pair of points order by thumb, index, middle, ring, pinky fingers
        bones_3d = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        edges_3d = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]

        ############ Display in 3D
        ax = fig.add_subplot((111), projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for b in bones_3d:
            ax.plot(copy_joints[b, 0], copy_joints[b, 1], copy_joints[b, 2], linewidth=1.0 , c='g')
        ax.scatter(copy_joints[:, 0], copy_joints[:, 1], copy_joints[:, 2],c="b", edgecolors='b')

        for e in edges_3d:
            ax.plot(copy_objCorners[e, 0], copy_objCorners[e, 1], copy_objCorners[e, 2], linewidth=1.0 , c='m')
        ax.scatter(copy_objCorners[:, 0], copy_objCorners[:, 1], copy_objCorners[:, 2],c="c", edgecolors='c')

    plt.ion()
    plt.draw()
    plt.pause(1)
    plt.waitforbuttonpress()
    plt.close('all')

def convertXYZToxyz(points, center):
    points = points - center
    points = points / cube_size
    return points

if __name__ == '__main__':
    train_folder = '/data/Suparna/Ho3D/train/'
    train_txt = '/data/Suparna/Ho3D/train.txt'

    # TrainfileWrite contains the path of hdf files being created
    # TrainfileWrite = open( '/data/Suparna/Ho3D_hd5/trainfiles.txt', 'a+')
    # # folder path to save data in hd5
    # dest_folder = '/data/Suparna/Ho3D_hd5/'
    # depth_h5, joint_22_h5, = [], []
    # cnt = 0
    # chunck = 0

    with open(train_txt) as tf:
        for record in tf:
            folder,file = tuple(record.rstrip().split('/'))
            depthpath = os.path.join(train_folder,folder,'depth',file+'.png')
            annotpath = os.path.join(train_folder,folder,'meta',file+'.pkl')
            depth = read_depth_img(depthpath)
            annot = np.load(annotpath, allow_pickle=True)
            camMat = annot['camMat']
            handJoints = annot['handJoints3D']
            objCorners = annot['objCorners3D']

            handJoints_uvd = project_3D_points(camMat, handJoints[jointsMapManoToSimple])
            obj_uvd = project_3D_points(camMat, objCorners)

            com = getCenter(handJoints_uvd, obj_uvd)
            print ('com:', com)

            ################## draw skeleton on orginal image ############
            #drawSkeleton(handJoints_uvd,obj_uvd,depth,com)

            cropped, transMat = CropImage(depth, com)
            norm_hand , norm_corner = normalizeAnnot(transMat,com,handJoints_uvd,obj_uvd)
            ################## draw skeleton on cropped image ############
            drawProcessedImg(norm_hand, norm_corner, cropped)

            ############### verify de-normalizing annotations - to be used after prediction #############
            denorm_hand , denorm_corner = deNormalizeAnnot(transMat,com,norm_hand,norm_corner)
            #drawSkeleton(denorm_hand, denorm_corner, depth, com)








