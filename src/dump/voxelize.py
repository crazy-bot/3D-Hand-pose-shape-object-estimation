import os
from util.main_util import *
from util.voxel_util import *


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z

def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y

def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points

def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels

def generate_pointcloud(depth, fx, fy, ux, uy,d):
    depth_scale = 0.00012498664727900177
    #depth = depth/(1000)
    h, w = depth.shape
    points = []
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            Z = depth[u,v]
            #print(Z)
            if (Z==0 or Z==1): continue
            #Z = (Z*cube_size)+d
            X = (u - w/2) * Z / fx
            Y = (v - h/2) * Z / fy
            points.append([X,Y,Z])
    return np.array(points)

def computeVoxel(points, size_x=None, size_y=None, size_z=None):
    xyzmin = points.min(0) # min value of x,y,z (1,3)
    xyzmax = points.max(0) #max value of x,y,z (1,3)
    sizes = [size_x, size_y, size_z]
    x_y_z = [1,1,1]
    for n, size in enumerate(sizes):
        if size is None:
            continue
        ########### numpy.ptp: Range of values (maximum - minimum) along an axis.
        margin = (((points.ptp(0)[n] // size) + 1) * size) - points.ptp(0)[n]
        xyzmin[n] -= margin / 2
        xyzmax[n] += margin / 2
        x_y_z[n] = ((xyzmax[n] - xyzmin[n]) / size).astype(int)

    segments = []
    shape = []
    for i in range(3):
        # note the +1 in num
        s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
        segments.append(s)
        shape.append(step)


def get_voxel(depth):
    voxel_size = 150
    voxel = np.zeros((150, 150, 150))
    depth = (depth+1)/2
    depth = depth * voxel_size
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            z = depth[u, v]
            if (z == 0 or z == voxel_size): continue
            current_point = (u, v, int(z))
            voxel[current_point] = 1

    return voxel

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def plot3dVisualize(ax, m, flip_x=False, c="b", alpha=0.1, camPose=np.eye(4, dtype=np.float32), isOpenGLCoords=False):
    '''
    Create 3D visualization
    :param ax: matplotlib axis
    :param m: mesh
    :param flip_x: flix x axis?
    :param c: mesh color
    :param alpha: transperency
    :param camPose: camera pose
    :param isOpenGLCoords: is mesh in openGL coordinate system?
    :return:
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if hasattr(m, 'r'):
        verts = np.copy(m.r)*1000
    elif hasattr(m, 'v'):
        verts = np.copy(m.v) * 1000
    else:
        raise Exception('Unknown Mesh format')
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    faces = np.copy(m.f)
    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        face_color = np.tile(np.array([[0., 0., 1., 1.]]), [verts.shape[0], 1])
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        face_color = np.tile(np.array([[1., 0., 0., 1.]]), [verts.shape[0], 1])
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()


if __name__ == '__main__':
    train_folder = '/data/Suparna/Ho3D/evaluation/'
    train_txt = '/data/Suparna/Ho3D/evaluation.txt'

    # TrainfileWrite contains the path of hdf files being created
    # TrainfileWrite = open( '/data/Suparna/Ho3D_hd5/trainfiles.txt', 'a+')
    # # folder path to save data in hd5
    # dest_folder = '/data/Suparna/Ho3D_hd5/'
    # depth_h5, joint_22_h5, = [], []
    # cnt = 0
    # chunck = 0
    voxelization_train = V2VVoxelization(cubic_size=200, augmentation=False)
    with open(train_txt) as tf:
        for record in tf:
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
            handJoints = annot['handJoints3D']
            handJoints = handJoints[jointsMapManoToSimple]
            objCorners = annot['objCorners3D']

            handJoints_uvd = project_3D_points(camMat, handJoints)
            obj_uvd = project_3D_points(camMat, objCorners)

            com = getCenter(handJoints_uvd, obj_uvd)
            print ('com:', com)

            ################## draw skeleton on orginal image ############
            #drawSkeleton(handJoints_uvd,obj_uvd,depth,com)

            # ################# Crop and segment hand-object from the whole image #################
            # cropped, transMat = CropImage(depth, com, fx, fy)
            # norm_hand, norm_corner = normalizeAnnot(transMat, com, handJoints_uvd.copy(), obj_uvd.copy())
            ################## draw skeleton on cropped image ############
            #drawProcessedImg(norm_hand, norm_corner, cropped)
            #
            # ############### verify de-normalizing annotations - to be used after prediction #############
            # denorm_hand = deNormalizeAnnot(transMat, com, cropped)
            #
            # drawProcessedImg(norm_hand, norm_corner, cropped)


#################### v2v approach : voxel segment ###############
            pointcloud = depthmap2points(depth,fx,fy)
            pointcloud = pointcloud.reshape(-1, 3)
            #np.savetxt('pointcloud.txt',pointcloud)
            h, w = depth.shape
            refpoint = pixel2world(com[0],com[1],com[2],w,h,fx,fy)
            joints_world = np.zeros((21, 3), dtype=np.float32)
            joints_world[:, 0], joints_world[ :, 1], joints_world[:, 2] = pixel2world(handJoints_uvd[:,0],handJoints_uvd[:,1],handJoints_uvd[:,2],w,h,fx,fy)
            corners_world = np.zeros((8, 3), dtype=np.float32)
            corners_world[:, 0], corners_world[:, 1], corners_world[:, 2] = pixel2world(obj_uvd[:, 0],
                                                                                     obj_uvd[:, 1],
                                                                                     obj_uvd[:, 2], w, h, fx, fy)
            sample = {
                'points': pointcloud,
                'joints': joints_world,
                'corners':corners_world,
                'refpoint': refpoint
            }
            voxel,heatmap_joints,heatmap_corners = voxelization_train(sample)

            ######### verify heatmap to joint conversion and plot on original image ################
            # jointsworld2 = voxelization_train.evaluate(heatmap_joints,refpoint)
            # jointsuvd2 = points2pixels(jointsworld2,w,h,fx,fy)
            # cornersworld2 = voxelization_train.evaluate(heatmap_corners, refpoint)
            # cornersuvd2 = points2pixels(cornersworld2, w, h, fx, fy)
            # drawSkeleton(jointsuvd2,cornersuvd2,depth,com)

            # voxel = get_voxel(cropped)
            # and plot voxel


            ######### voxel as matplotlib voxel #########
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # ax.voxels(voxel,facecolors='r',  edgecolor='b')
            # plt.show()

            ######### voxel as pointcloud #########
            coord_x = np.argwhere(voxel)[:,0]
            coord_y = np.argwhere(voxel)[:,1]
            coord_z = np.argwhere(voxel)[:,2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.scatter(coord_x,coord_y,coord_z,c='r')

            ################ handling mesh ###############
            _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])
            print('handmesh ',handMesh.shape)
            plot3dVisualize(ax, handMesh, flip_x=False, isOpenGLCoords=True, c="r")
            plt.show()







