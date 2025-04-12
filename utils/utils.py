from functools import partial
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import os
import errno

def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c

def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0,1,3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im
def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im

def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

def homography_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :] # E^-1 = [R^T, -R^T*t]
    H_g2c = E_inv[:, [0,1,3]] # H_g2c = [R^T[:,0] R^T[:,1] -R^T*t]
    H_g2im = np.matmul(K, H_g2c) # H_g2im = K * H_g2c
    return H_g2im

def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :] # E^-1 = [R^T, -R^T*t]
    P_g2im = np.matmul(K, E_inv)    # P_g2im = K * E_inv
    return P_g2im

def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im


def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im


def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """
    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = self.alpha*((1 - pt) ** self.gamma)* ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss
    
def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d