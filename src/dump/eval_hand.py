from config import *
import sys
sys.path.append(PROJECT_PATH)
import json
from util.eval_util import EvalUtil
import os
import numpy as np
from util.main_util import forwardKinematics


folder_path = DATA_DIR + '/' + evalset
file_path = DATA_DIR + '/' + evalset + '.txt'


with open(handJsonPath) as fp:
    data = json.load(fp)
    pred_xyz = data[0]
    pred_verts = data[1]
    print('total xyz: ',len(pred_xyz))
    print('total verts: ', len(pred_verts))

# init eval utils
eval_xyz = EvalUtil()
eval_mesh_err = EvalUtil(num_kp=2358)

with open(file_path) as fp:
    files = fp.readlines()

    for i in range(len(files)):
        record = files[i]
        print(record)
        folder, file = tuple(record.rstrip().split('/'))
        annotpath = os.path.join(folder_path, folder, 'meta', file + '.pkl')
        annot = np.load(annotpath, allow_pickle=True)
        handJoints = annot['handJoints3D']
        objCorners = annot['objCorners3D']
        _, handMesh = forwardKinematics(annot['handPose'], annot['handTrans'], annot['handBeta'])

        xyz, verts = handJoints, handMesh.r
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        if(len(pred_xyz) > 0):
            xyz_pred = pred_xyz[i]
            xyz_pred = [np.array(x) for x in [xyz_pred]]
            # Not aligned errors
            eval_xyz.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred
            )
        if (len(pred_verts) > 0):
            verts_pred = pred_verts[i]
            verts_pred = [np.array(x) for x in [verts_pred]]
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

    # Calculate results
    print('completed ',handJsonPath)
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D MESH results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))