from __future__ import print_function, unicode_literals
from config import *
import sys

sys.path.append(PROJECT_PATH)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pip
import argparse
import json


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])


try:
    import open3d as o3d
except:
    install('open3d-python')
    import open3d as o3d

try:
    from scipy.linalg import orthogonal_procrustes
except:
    install('scipy')
    from scipy.linalg import orthogonal_procrustes

from util.vis_utils import *
from util.eval_util import EvalUtil


def verts2pcd(verts, color=None):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr)  # closest dist for each gt point
    d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt)  # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(
            len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


def align_sc_tr(mtx1, mtx2):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = mtx2.copy()
    # normalize the predictions
    s = np.sqrt(np.sum(np.square(predCurr[9] - predCurr[0])))
    if s > 0:
        predCurr = predCurr / s

    # get the scale of the ground truth
    sGT = np.sqrt(np.sum(np.square(mtx1[9] - mtx1[0])))

    # make predictions scale same as ground truth scale
    predCurr = predCurr * sGT

    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_sc_tr_al = predCurrRel + mtx1[0:1, :]

    return preds_sc_tr_al


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    curve_data_list = list()
    for i,item in enumerate(curve_list):
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path_{}.png".format(i))
        plt.savefig(img_path, bbox_inches=0, dpi=300)

    #     # write image and create html embedding
    #     data_uri1 = open(img_path, 'rb').read().encode('base64').replace('\n', '')
    #     img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
    #     curve_data_list.append((item.text, img_tag1))
    #
    #     os.remove(img_path)
    #
    # htmlString = '''<!DOCTYPE html>
    # <html>
    # <body>
    # <h1>Detailed results:</h1>'''
    #
    # for i, (text, img_embed) in enumerate(curve_data_list):
    #     htmlString += '''
    #     <h2>%s</h2>
    #     <p>
    #     <img border="0" %s alt="FROC" width="576pt" height="432pt">
    #     </p>
    #     <p>Raw curve data:</p>
    #
    #     <p>x_axis: <small>%s</small></p>
    #     <p>y_axis: <small>%s</small></p>
    #
    #     ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)
    #
    # htmlString += '''
    # </body>
    # </html>'''
    #
    # htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    # htmlfile.write(htmlString)
    # htmlfile.close()


def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception('Giving up, because its not clear which file to evaluate.')


def main(output_dir):
    # init eval utils
    eval_xyz, eval_xyz_procrustes_aligned, eval_xyz_sc_tr_aligned = EvalUtil(num_kp=8), EvalUtil(num_kp=8), EvalUtil(
        num_kp=8)
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=2358), EvalUtil(num_kp=2358)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]

    shape_is_mano = True

    with open(objJsonPath) as fp:
        data = json.load(fp)
        pred_xyz = data[0]
        pred_verts = data[1]
        print('total xyz: ', len(pred_xyz))
        print('total verts: ', len(pred_verts))

    with open(file_path) as fp:
        files = fp.readlines()

    # iterate over the dataset once
    for i in range(len(files)):
        record = files[i]
        print(record)
        folder, file = tuple(record.rstrip().split('/'))
        annotpath = os.path.join(folder_path, folder, 'meta', file + '.pkl')
        annot = np.load(annotpath, allow_pickle=True)
        objMesh = read_obj(os.path.join(OBJ_MODEL_PATH, annot['objName'], 'textured_2358.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(annot['objRot'])[0].T) + annot['objTrans']
        objCorners = annot['objCorners3D']

        xyz, verts = objCorners, objMesh.v
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred = pred_xyz[i]
        xyz_pred = [np.array(x) for x in [xyz_pred]][0]

        verts_pred = pred_verts[i]
        verts_pred = [np.array(x) for x in [verts_pred]][0]

        #Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )

        if shape_is_mano:
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

        # F-scores
        l, la = list(), list()
        for t in f_threshs:
            # for each threshold calculate the f score and the f score of the aligned vertices
            f, _, _ = calculate_fscore(verts, verts_pred, t)
            l.append(f)
        f_score.append(l)

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    if shape_is_mano:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

    print('F-scores')
    f_out = list()
    f_score = np.array(f_score).T
    for f, t in zip(f_score, f_threshs):
        print('F@%.1fmm = %.3f' % (t * 1000, f.mean()))
        f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        xyz_mean3d *= 100
        fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
        fo.write('xyz_auc3d: %f\n' % xyz_auc3d)

        mesh_mean3d *= 100
        fo.write('mesh_mean3d: %f\n' % mesh_mean3d)
        fo.write('mesh_auc3d: %f\n' % mesh_auc3d)
        for t in f_out:
            fo.write('%s\n' % t)
    print('Scores written to: %s' % score_path)

    createHTML(
        output_dir,
        [
            curve(thresh_xyz * 100, pck_xyz, 'Distance in cm', 'Percentage of correct keypoints',
                  'PCK curve for not aligned keypoint error'),
            curve(thresh_mesh * 100, pck_mesh, 'Distance in cm', 'Percentage of correct vertices',
                  'PCV curve for mesh error'),
        ]
    )

    print('Evaluation complete.')


folder_path = DATA_DIR + '/' + evalset
file_path = DATA_DIR + '/' + evalset + '.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to where the eval result should be.')

    args = parser.parse_args()

    # call eval
    main(
        args.output_dir,
    )
