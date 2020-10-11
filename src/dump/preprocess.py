import os
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

fx = 614.62700
fy = 614.10100
#ux =
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


def CropImage(image,com,joints):
  cube_size = 150
  img_size = 150

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

  joints[:,2] = 1
  trans_jonts = np.dot(trans,joints.T)
  trans_jonts /= cube_size
  return res_img,trans_jonts.T

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

def normalizeImage(img):
    img = img.astype(np.float32)  # convert to float
    img -= img.min()
    img /= img.max()
    #img = np.uint8((img) * 255)
    return  img

#################### draw skeleton and object bounding box
def drawSkeleton(joints,objCorners,img, mode='2D'):
    fig = plt.figure()
    palmcenter = getPalmCenter(joints)
    objcenter = getObjCenter(objCorners)
    print('palmcenter ',palmcenter)
    print('objcenter ', objcenter)
    joints = np.append(joints, [palmcenter], axis=0)

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

    plt.ion()
    plt.draw()
    plt.pause(1)
    plt.waitforbuttonpress()
    plt.close('all')
    #return plt

def drawProcessedImg(joints,img ):
    #J = joints.shape[0]
    img_scaled = (img + 1) / 2
    im = np.zeros((img.shape[0], img.shape[1], 3))
    im[:, :, 0] = img_scaled
    im[:, :, 1] = img_scaled
    im[:, :, 2] = img_scaled

    joints[:, 0] = (joints[:, 0] +1) / 2 * img.shape[0]
    joints[:, 1] = (joints[:, 1]+1) / 2 * img.shape[1]

    # joints[:, 0] = joints[:, 0]  * img.shape[0]
    # joints[:, 1] = joints[:, 1] * img.shape[1]

    bones = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],[0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    # skeleton array to hold the x,y coordinates
    ske_joint = []
    for i in range(0, len(bones)):
        pairset = bones[i]
        X = []
        Y = []
        for j in pairset:
            X.append(joints[j][0])
            Y.append(joints[j][1])
            ske_joint.append([X, Y])

    #print  joints

    # plt.close() will close the figure window entirely, where plt.clf() will just clear the figure - you can still paint another plot onto it.
    plt.clf()
    plt.imshow(img_scaled, 'gray')
    plt.scatter(joints[:, 0], joints[:, 1], s=30, c='r', edgecolors='r', alpha=0.5)

    for s in ske_joint:
        plt.plot(s[0], s[1], linewidth=2.0, c='y')

    #plt.ion()
    #plt.show()
    plt.waitforbuttonpress()
    plt.axis('off')
    #plt.pause(0.000000000001)

def convertXYZToxyz(points, center):
    points = points - center
    points = points / 150
    return points

if __name__ == '__main__':
    depthpath = '/data/Guha/TestDepthSynth/ABF10/depth'
    rgbpath = '/data/Guha/TestDepthSynth/ABF10/rgb'
    metapath = '/data/Guha/TestDepthSynth/ABF10/meta'
    depthimglist = sorted(os.listdir(depthpath))
    metalist = sorted(os.listdir(metapath))

    camMat = np.load(metapath + '/' + metalist[0], allow_pickle=True)['camMat']
    fx = camMat[0, 0]
    fy = camMat[1, 1]
    ux = camMat[0, 2]
    uy = camMat[1, 2]
    for i in range(1000,len(depthimglist)):
        depth = read_depth_img(depthpath+'/'+depthimglist[i])

        annot = np.load(metapath+'/'+metalist[i],allow_pickle=True)
        handJoints = annot['handJoints3D']
        objCorners = annot['objCorners3D']

        handJoints_uvd = project_3D_points(camMat,handJoints[jointsMapManoToSimple])
        obj_uvd = project_3D_points(camMat, objCorners, is_OpenGL_coords=True)
        #drawSkeleton(handJoints_uvd,obj_uvd,depth)
        com = getCenter(handJoints_uvd,obj_uvd)
        print ('com:',com)
        cropped,trans = CropImage(depth,com,handJoints_uvd)

        ########### display for testing ###############
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.imshow(cropped,'gray')
        # ax1.title.set_text('after crop')
        # plt.show()
        # palmcenter = getPalmCenter(handJoints_uvd)
        # norm_joints = convertXYZToxyz(handJoints_uvd,com)
        # drawProcessedImg(norm_joints,cropped)


