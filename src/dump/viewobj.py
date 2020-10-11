from dataset.HO_Data.vis_util import *
from dataset.HO_Data.data_util import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d
from util.contactloss import compute_contact_loss
import torch

mode = 'train'
train_folder = '/data/Guha/HO3D/{}/'.format(mode)
train_txt = '/data/Guha/HO3D/{}.txt'.format('valid')
with open(train_txt) as tf:
    for record in tf:
        print(record)
        folder, file = tuple(record.rstrip().split('/'))
        annotpath = os.path.join(train_folder, folder, 'meta', file + '.pkl')
        annot = np.load(annotpath, allow_pickle=True)
        camMat = annot['camMat']
        fx = camMat[0, 0]
        fy = camMat[1, 1]
        ux = camMat[0, 2]
        uy = camMat[1, 2]
        _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])
        print('handmesh ', handMesh.shape)
        handMesh_uvd = project_3D_points(camMat, handMesh)
        objMesh = read_obj(os.path.join('/data/Guha/YCB_Video_Models/', 'models', annot['objName'], 'textured_2358.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']
        print('objMesh ', objMesh.v.shape)
        objmesh_uvd = project_3D_points(camMat, objMesh.v)

        mesh = open3d.io.read_triangle_mesh(
            os.path.join('/data/Guha/YCB_Video_Models/', 'models', '025_mug', 'textured_4500.obj'))
        handmesh_world = Main_pixelToworld(handMesh_uvd.copy(), ux, uy, fx, fy)
        objmesh_world = Main_pixelToworld(objmesh_uvd.copy(), ux, uy, fx, fy)
        #open3d.visualization.draw_geometries([objMesh])
        #open3dVisualize([handMesh,objMesh],['r','g'])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        handverts = plot3dVisualize(ax, handMesh, verts=handmesh_world, flip_x=False, isOpenGLCoords=False, c="r")
        objverts = plot3dVisualize(ax, objMesh, verts=objmesh_world, flip_x=False, isOpenGLCoords=False, c="b")
        allverts = np.vstack([handverts,objverts])
        ax.set_xlim(min(allverts[:,0]),max(allverts[:,0]))
        ax.set_ylim(min(allverts[:,1]),max(allverts[:,1]))
        ax.set_zlim(min(allverts[:,2]),max(allverts[:,2]))
        plt.show()

        a = torch.from_numpy(handmesh_world.astype(np.float32)).unsqueeze(0)
        b = torch.from_numpy(handMesh.f.astype(int))
        c = torch.from_numpy(objmesh_world.astype(np.float32)).unsqueeze(0)
        d = torch.from_numpy(objMesh.f.astype(int))
        penetr_loss, contact_info, metrics = compute_contact_loss(a, b, c, d)
        print(penetr_loss)

