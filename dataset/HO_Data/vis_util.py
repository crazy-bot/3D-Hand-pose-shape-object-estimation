import numpy as np
import open3d
import matplotlib.pyplot as plt

def testVis(voxel, joints, objCorners, norm_handmesh, norm_objmesh):
    import matplotlib.pyplot as plt

    coord_x = np.argwhere(voxel)[:, 0]
    coord_y = np.argwhere(voxel)[:, 1]
    coord_z = np.argwhere(voxel)[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(coord_x, coord_y, coord_z, c='r', s=10)

    ax.scatter(norm_handmesh[:, 0], norm_handmesh[:, 1], norm_handmesh[:, 2], c='green', s=30)
    ax.scatter(norm_objmesh[:, 0], norm_objmesh[:, 1], norm_objmesh[:, 2], c='pink', s=30)

    draw3dpose(ax,joints,objCorners)
    plt.show()

def testVis2(voxel, joints, objCorners):
    import matplotlib.pyplot as plt

    coord_x = np.argwhere(voxel)[:, 0]
    coord_y = np.argwhere(voxel)[:, 1]
    coord_z = np.argwhere(voxel)[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(coord_x, coord_y, coord_z, c='r', s=10)

    draw3dpose(ax,joints,objCorners)
    plt.show()

def testVis3(voxel):
    import matplotlib.pyplot as plt

    coord_x = np.argwhere(voxel)[:, 0]
    coord_y = np.argwhere(voxel)[:, 1]
    coord_z = np.argwhere(voxel)[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #ax.scatter(coord_x, coord_y, coord_z, c='r', s=10)
    ax.voxels(voxel[0],facecolors='r',alpha=0.4)
    ax.axis('off')
    #ax.scatter(ref[:,0], ref[:,1], ref[:,2], c='b', s=10)
    #draw3dpose(ax,joints,objCorners)
    ax.view_init(elev=-60, azim=0)
    plt.show()


def draw3dpose(ax,joints,objCorners):
    ############## bone joints indexe as pair of points order by thumb, index, middle, ring, pinky fingers
    bones_3d = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    edges_3d = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    hand_points = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[0]]
    obj_points = []
    ############ Display in 3D
    colors = ['r','g','m','b','c','black']
    for i,hp in enumerate(hand_points):
        ax.scatter(joints[hp, 0], joints[hp, 1], joints[hp, 2], c=colors[i], s=100)

    for b in bones_3d:
        ax.plot(joints[b, 0], joints[b, 1], joints[b, 2], linewidth=2.0, c='r')
    #ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=joints[:, 2], edgecolors='b', s=30)

    for e in edges_3d:
        ax.plot(objCorners[e, 0], objCorners[e, 1], objCorners[e, 2], linewidth=2.0, c='m')
    ax.scatter(objCorners[:, 0], objCorners[:, 1], objCorners[:, 2], c='saddlebrown', s=100)
    
    ax.view_init(elev=-70, azim=-90)
    ax.axis('off')


def open3dVisualize(mList, colorList):

    o3dMeshList = []
    for i, m in enumerate(mList):
        mesh = open3d.geometry.TriangleMesh()
        numVert = 0
        if hasattr(m, 'r'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.r))
            numVert = m.r.shape[0]
        elif hasattr(m, 'v'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.v))
            numVert = m.v.shape[0]
        else:
            raise Exception('Unknown Mesh format')
        # verts = mesh.vertices
        # extra = np.zeros((6,3))
        # verts = np.vstack([np.asarray(verts),extra])
        # mesh.vertices = open3d.utility.Vector3dVector(np.copy(verts))
        # numVert = len(verts)
        mesh.triangles = open3d.utility.Vector3iVector(np.copy(m.f))
        if colorList[i] == 'r':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
        elif colorList[i] == 'g':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
        else:
            raise Exception('Unknown mesh color')

        o3dMeshList.append(mesh)
    open3d.visualization.draw_geometries(o3dMeshList)

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

def plot3dVisualize(ax, m, verts, flip_x=False, c="b", alpha=0.1, camPose=np.eye(4, dtype=np.float32), isOpenGLCoords=False):
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
    # if hasattr(m, 'r'):
    #     verts = np.copy(m.r)*1000
    # elif hasattr(m, 'v'):
    #     verts = np.copy(m.v) * 1000
    # else:
    #     raise Exception('Unknown Mesh format')

    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    faces = np.copy(m.f)
    ax.view_init(elev=10, azim=-50)
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
    #cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    #plt.tight_layout()
    return verts

#################### draw skeleton and object bounding box on original image #################
def plotOnOrgImg(ax ,joints,objCorners,img):
    objcenter = np.mean(objCorners, axis=0)
    com = np.mean(np.array([joints[0], objcenter]), axis=0)

    # bone joints indexe as set of points order by thumb, index, middle, ring, pinky fingers
    bones = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
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
    ax.imshow(img, 'gray')
    ax.scatter(joints[:, 0], joints[:, 1], s=30, c='y', edgecolors='y', alpha=0.5)
    # ax.scatter(palmcenter[0], palmcenter[1], s=30, c='r')

    ax.scatter(objCorners[:, 0], objCorners[:, 1], s=30, c='c')
    # plt.scatter(objcenter[0], objcenter[1], s=30, c='r')

    plt.scatter(com[0], com[1], s=30, c='r')

    ############ draw the bones
    for s in ske_joint:
        plt.plot(s[0], s[1], linewidth=1.0, c='g')
    for o in obj_edges:
        plt.plot(o[0], o[1], linewidth=1.0, c='m')

    return plt

def plot2DforTest(depth,hand_keypoints_uvd, objbboxUVD,handmesh_uvd,objmesh_uvd,fileName):
    fig = plt.figure(figsize=(30, 30))
    # show Prediction
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(depth, 'gray')
    ax1.scatter(handmesh_uvd[:, 0], handmesh_uvd[:, 1], c=handmesh_uvd[:, 2], s=15)
    ax1.scatter(objmesh_uvd[:, 0], objmesh_uvd[:, 1], c=objmesh_uvd[:, 2], s=15)
    plotOnOrgImg(ax1, hand_keypoints_uvd, objbboxUVD, depth)
    ax1.title.set_text('Prediction')
    plt.savefig(fileName)
    plt.close()

def plot2DforValid(depth,gtUVD,outUVD,fileName):
    fig = plt.figure(figsize=(30, 30))
    # show Prediction
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow(depth, 'gray')
    ax0.scatter(gtUVD['handmeshUVD'][:, 0], gtUVD['handmeshUVD'][:, 1], c=gtUVD['handmeshUVD'][:, 2], s=10)
    ax0.scatter(gtUVD['objmeshUVD'][:, 0], gtUVD['objmeshUVD'][:, 1], c=gtUVD['objmeshUVD'][:, 2], s=10)
    plotOnOrgImg(ax0, gtUVD['hand_uvd'], gtUVD['objbboxUVD'], depth)
    ax0.title.set_text('Ground Truth')

    # show Prediction
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(depth, 'gray')
    ax1.scatter(outUVD['handmeshUVD'][:, 0], outUVD['handmeshUVD'][:, 1], c=outUVD['handmeshUVD'][:, 2], s=10)
    ax1.scatter(outUVD['objmeshUVD'][:, 0], outUVD['objmeshUVD'][:, 1], c=outUVD['objmeshUVD'][:, 2], s=10)
    plotOnOrgImg(ax1, outUVD['hand_uvd'], outUVD['objbboxUVD'], depth)
    ax1.title.set_text('Prediction')

    plt.savefig(fileName)
    plt.close()

def plot3DPCforValid(gtUVD,outUVD,fileName):
    fig = plt.figure(figsize=(30, 30))
    # show Prediction
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.view_init(elev=0, azim=-50)
    ax0.scatter(gtUVD['handmeshUVD'][:, 0], gtUVD['handmeshUVD'][:, 1], gtUVD['handmeshUVD'][:, 2], c=gtUVD['handmeshUVD'][:, 2], s=10)
    ax0.scatter(gtUVD['objmeshUVD'][:, 0], gtUVD['objmeshUVD'][:, 1], gtUVD['objmeshUVD'][:, 2], c=gtUVD['objmeshUVD'][:, 2], s=10)
    draw3dpose(ax0, gtUVD['hand_uvd'], gtUVD['objbboxUVD'])
    ax0.title.set_text('Ground Truth')

    ##### show Prediction
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.view_init(elev=0, azim=-50)
    ax1.scatter(outUVD['handmeshUVD'][:, 0], outUVD['handmeshUVD'][:, 1], outUVD['handmeshUVD'][:, 2], c=outUVD['handmeshUVD'][:, 2], s=10)
    ax1.scatter(outUVD['objmeshUVD'][:, 0], outUVD['objmeshUVD'][:, 1], outUVD['objmeshUVD'][:, 2], c=outUVD['objmeshUVD'][:, 2], s=10)
    draw3dpose(ax1, outUVD['hand_uvd'], outUVD['objbboxUVD'])
    ax1.title.set_text('Prediction')
    plt.savefig(fileName)
    plt.close()

def plot3DPCforTest(outUVD,fileName):
    ##### show Prediction
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.view_init(elev=-70, azim=-90)
    ax1.scatter(outUVD['handmeshUVD'][:, 0], outUVD['handmeshUVD'][:, 1], outUVD['handmeshUVD'][:, 2], c='r', s=10)
    ax1.scatter(outUVD['objmeshUVD'][:, 0], outUVD['objmeshUVD'][:, 1], outUVD['objmeshUVD'][:, 2], c='b', s=10)
    draw3dpose(ax1, outUVD['hand_uvd'], outUVD['objbboxUVD'])
    ax1.set_title('Prediction', pad=10)
    ax1.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig(fileName)
    plt.close()
    #plt.show()

def plot3DMeshforValid(handMesh,objMesh, gtUVD,outUVD,fileName):
    fig = plt.figure()
    # show Prediction
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.view_init(elev=-70, azim=-90)
    ax0.axis('off')
    handverts = plot3dVisualize(ax0, handMesh, verts=gtUVD['handmeshUVD'], flip_x=False, isOpenGLCoords=False, c="r")
    objverts =plot3dVisualize(ax0, objMesh, verts=gtUVD['objmeshUVD'], flip_x=False, isOpenGLCoords=False, c="b")
    ax0.title.set_text('Ground Truth')
    allverts = np.vstack([handverts, objverts])
    ax0.set_xlim(min(allverts[:, 0]), max(allverts[:, 0]))
    ax0.set_ylim(min(allverts[:, 1]), max(allverts[:, 1]))
    ax0.set_zlim(min(allverts[:, 2]), max(allverts[:, 2]))

    ##### show Prediction
    #fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.view_init(elev=-70, azim=-90)
    ax1.axis('off')
    handverts = plot3dVisualize(ax1, handMesh, verts=outUVD['handmeshUVD'], flip_x=False, isOpenGLCoords=False, c="r")
    objverts = plot3dVisualize(ax1, objMesh, verts=outUVD['objmeshUVD'], flip_x=False, isOpenGLCoords=False, c="b")
    ax1.title.set_text('Prediction')
    allverts = np.vstack([handverts, objverts])
    ax1.set_xlim(min(allverts[:, 0]), max(allverts[:, 0]))
    ax1.set_ylim(min(allverts[:, 1]), max(allverts[:, 1]))
    ax1.set_zlim(min(allverts[:, 2]), max(allverts[:, 2]))
    #plt.show()
    plt.savefig(fileName)
    plt.close()

def plot3DMeshforTest(handmodel,objmodel,handMesh,objMesh,fileName):
    fig = plt.figure()
    # show Prediction
    ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    ax0.view_init(elev=-70, azim=-90)
    ax0.axis('off')
    handverts = plot3dVisualize(ax0, handmodel, verts=handMesh, flip_x=False, isOpenGLCoords=False, c="r")
    objverts =plot3dVisualize(ax0, objmodel, verts=objMesh, alpha=.02, flip_x=False, isOpenGLCoords=False, c="b")
    ax0.title.set_text('Ground Truth')
    allverts = np.vstack([handverts,objverts])
    ax0.set_xlim(min(allverts[:, 0]), max(allverts[:, 0]))
    ax0.set_ylim(min(allverts[:, 1]), max(allverts[:, 1]))
    ax0.set_zlim(min(allverts[:, 2]), max(allverts[:, 2]))
    plt.savefig(fileName)
    plt.close()
