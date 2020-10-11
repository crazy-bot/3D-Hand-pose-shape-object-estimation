from dataset.HO_Data.data_util import *
import json
coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def pred2Org_handjoints(heatmap,refpoint,ux, uy, fx, fy):
    out_handUVD = evaluate(heatmap, refpoint, cubic_size=200, cropped_size=88, pool_factor=2)
    out_handUVD = Main_worldTopixel(out_handUVD, ux, uy, fx, fy)
    out_handUVD[:, 2] = out_handUVD[:, 2] / 1000
    handxyz = Main_pixelToworld(out_handUVD.copy(), ux, uy, fx, fy)
    handxyz = handxyz[jointsMapSimpleToMano]
    handxyz = handxyz.dot(coord_change_mat.T)

    return out_handUVD,handxyz


def pred2Org_objbbox(heatmap, refpoint, ux, uy, fx, fy):
    out_objUVD = evaluate(heatmap, refpoint, cubic_size=200, cropped_size=88,pool_factor=2)
    out_objUVD = Main_worldTopixel(out_objUVD, ux, uy, fx, fy)
    out_objUVD[:, 2] = out_objUVD[:, 2] / 1000
    objxyz = Main_pixelToworld(out_objUVD.copy(), ux, uy, fx, fy)
    objxyz = objxyz.dot(coord_change_mat.T)

    return out_objUVD, objxyz


def pred2Org_mesh(out_mesh, refpoint, ux, uy, fx, fy):
    out_meshUVD = warp2continuous(out_mesh, refpoint, 200, 44)
    out_meshUVD = Main_worldTopixel(out_meshUVD, ux, uy, fx, fy)
    out_meshUVD[:, 2] = out_meshUVD[:, 2] / 1000
    verts = Main_pixelToworld(out_meshUVD.copy(), ux, uy, fx, fy)
    verts = verts.dot(coord_change_mat.T)

    return out_meshUVD, verts

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

def calculateBatchLoss(out, GT):
    mean_Hpose, mean_Opose = calculatePoseBatchLoss(out, GT)
    mean_Hmesh, mean_Omesh = calculateMeshBatchLoss(out,GT)
    return mean_Hpose,mean_Opose,mean_Hmesh,mean_Omesh

def calculatePoseBatchLoss(out, GT):
    mean_Hpose,mean_Opose = [],[]

    out_handpose = out['handpose'].cpu().numpy()
    out_objpose = out['objpose'].cpu().numpy()

    GT_handpose = GT['handpose'].cpu().numpy()
    GT_objpose = GT['objpose'].cpu().numpy()

    camMat = GT['camMat'].cpu().numpy()
    refpoint = GT['refpoint'][0].cpu().numpy()

    batch_sz = out['handpose'].size()[0]
    for i in range(batch_sz):
        fx = camMat[i, 0, 0]
        fy = camMat[i, 1, 1]
        ux = camMat[i, 0, 2]
        uy = camMat[i, 1, 2]
        _, handxyz = pred2Org_handjoints(out_handpose[i], refpoint, ux, uy, fx, fy)
        mean_Hpose.append(np.mean(np.linalg.norm((GT_handpose[i] - handxyz), axis=1)))

        _, objxyz = pred2Org_objbbox(out_objpose[i], refpoint, ux, uy, fx, fy)
        mean_Opose.append(np.mean(np.linalg.norm((GT_objpose[i] - objxyz),axis=1)))

    mean_Hpose = np.mean(np.asarray(mean_Hpose))
    mean_Opose = np.mean(np.asarray(mean_Opose))

    return mean_Hpose,mean_Opose

def calculateMeshBatchLoss(out, GT):
    mean_Hmesh, mean_Omesh = [], []

    out_handverts = out['handverts'].cpu().numpy()
    out_objverts = out['objverts'].cpu().numpy()

    GT_handverts = GT['handverts'].cpu().numpy()
    GT_objverts = GT['objverts'].cpu().numpy()
    camMat = GT['camMat'].cpu().numpy()
    refpoint = GT['refpoint'][0].cpu().numpy()

    batch_sz = out['handverts'].size()[0]
    for i in range(batch_sz):
        fx = camMat[i, 0, 0]
        fy = camMat[i, 1, 1]
        ux = camMat[i, 0, 2]
        uy = camMat[i, 1, 2]

        _, handverts = pred2Org_mesh(out_handverts[i], refpoint, ux, uy, fx, fy)
        mean_Hmesh.append(np.mean(np.linalg.norm((GT_handverts[i] - handverts),axis=1)))

        _, objverts = pred2Org_mesh(out_objverts[i], refpoint, ux, uy, fx, fy)
        mean_Omesh.append(np.mean(np.linalg.norm((GT_objverts[i] - objverts),axis=1)))

    mean_Hmesh = np.mean(np.asarray(mean_Hmesh))
    mean_Omesh = np.mean(np.asarray(mean_Omesh))

    return mean_Hmesh,mean_Omesh


