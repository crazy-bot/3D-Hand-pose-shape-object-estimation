import os
import numpy as np
import cv2
from src.path import MANO_MODEL_PATH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#################### mapping constant #######################
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
jointsMapSimpleToMano = [0,
                         5, 6, 7, 9,
                         10, 11, 17, 18,
                         19, 13, 14, 15,
                         1, 2, 3, 4,
                         8, 12, 16, 20]

####################### utility functions for coordinate system conversion ################

def Main_pixelToworld(x, ux, uy, fx, fy,):
    x[ :, 0] = (x[ :, 0] - ux) * x[ :, 2] / fx
    x[ :, 1] = (x[ :, 1] - uy) * x[ :, 2] / fy
    return x

def Main_worldTopixel(x, ux, uy,fx, fy,):
    x[:, 0] = x[ :, 0] * fx / x[ :, 2] + ux
    x[ :, 1] = x[ :, 1] * fy / x[ :, 2] + uy
    #return x.astype(int)
    return x

def Main_depthPixelToworld(x, y, image, ux, uy, fx, fy):
    x = (x - ux) * image /fx
    y = (y - uy) * image / fy
    return x,y,image

def Main_depthmap2points(image, ux, uy, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = Main_depthPixelToworld(x, y, image, ux, uy, fx, fy)
    return points

####################### utility functions for coordinate system conversion: taken from V2V ################

def v2vpixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def v2vworld2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y


def v2vdepthmap2points(image, ux, uy, fx, fy ):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = v2vpixel2world(x, y, image, ux, uy, fx, fy)
    return points


def v2vpoints2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        v2vworld2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels

####################### I/O functions for HO3D data ###################

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""
    #depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    #_assert_exist(depth_filename)
    #print (depth_filename)
    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256

    dpt = dpt * depth_scale * 1000
    #dpt = dpt * depth_scale
    dpt[dpt==0] = 2000
    return dpt

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


if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model



def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            #if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])


    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result

def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 66034  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 11524
    else:
        assert 0, 'Invalid choice.'
######################## preprocessing and post processing utility ################################
###################################################################################################
def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size  + refpoint

    return coord


def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic


def extract_coord_from_output(output, center=True):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    center: if True, add 0.5, default is true
    return: shape (batch, jointNum, 3)
    '''
    assert (len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T

    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    # Note discrete coord can represents real range [coord, coord+1), see function scattering()
    # So, move coord to range center for better fittness
    if center: xyz_output = xyz_output + 0.5


    return xyz_output


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize
    #coord = (coord - refpoint) / (cubic_size/2)  # -> [-1, 1] ########### existing
    coord = (coord - refpoint) / (cubic_size) ############ modification

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2)  # move center to original volume

    # resize around original volume center
    resize_scale = new_size / 100
    if new_size < 100:
        coord = coord * resize_scale + original_size/2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size/2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,0] -= original_size / 2
        original_coord[:,1] -= original_size / 2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += original_size / 2
        coord[:,1] += original_size / 2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode.
    coord -= trans

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord_depth = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord_depth, cropped_size)

    return cubic


def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

    # use center of cell
    center_offset = 0.5

    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x+center_offset-xi)/std, 2)/2 + \
            np.power((d3output_y+center_offset-yi)/std, 2)/2 + \
            np.power((d3output_z+center_offset-zi)/std, 2)/2))

    return heatmap


def evaluate(heatmaps, refpoints, cubic_size, cropped_size,pool_factor):
    coords = extract_coord_from_output(heatmaps)
    coords *= pool_factor
    keypoints = warp2continuous(coords, refpoints, cubic_size, cropped_size)
    return keypoints
