import os
import math
import torch
import shutil
import numpy as np


def get_shmpath():
    # this can be replaced with a path on /dev/shm for faster reading / writing if you know what you're doing
    shmpath = 'results/shm_dummy'

    if os.path.exists(shmpath):
        shutil.rmtree(shmpath)  # remove old imgs

    if not os.path.exists(shmpath):
        os.makedirs(shmpath)
    return shmpath


def set_initial_and_gt(hparams, seed=0):
    device = hparams['device']
    ndim = 28 * 12
    init = torch.rand([ndim], device=device, requires_grad=True)
    gt = torch.rand([ndim], device=device)

    hparams['ndim'] = ndim
    hparams['theta'] = init
    hparams['gt_theta'] = gt


def get_defaults(mode):
    objects = []

    primary_objects = ['00', '01', '02', '03', '04', '05', '06',
                       '10', '11', '12', '13', '14', '15', '16',
                       '20', '21', '22', '23', '24', '25', '26',
                       '30', '31', '32', '33', '34', '35', '36']

    for o in primary_objects:
        objects.append(o)
        for arr in range(1, 12):
            objects.append(o + '.0' + '{:02d}'.format(arr))

    defaults = {k: 1 for k in objects}
    defaults['id'] = -1
    #defaults['iter'] = -1
    return defaults


# blender wants a string that tells it for every element whether the #el led at idx #idx is #on or #off
def update_binary(valdict, b):
    # expects b to be a list or tensor with 28 entries that will be converted to [0,1]
    if isinstance(b, torch.Tensor): b = b.squeeze().tolist()
    if isinstance(b, np.ndarray): b = list(b)
    assert len(b) % 28 == 0

    num_entries = int(len(b) / 28)
    for outer_idx in range(num_entries):    # loop through all cell entries for highdim, for regular led exp, this is 1
        currlist = b[outer_idx:outer_idx+28]
        for idx in range(28):
            idx_str = '' if outer_idx == 0 else '.0{:02d}'.format(outer_idx)          # ident for highdim, e.g., .008
            valdict[f'{math.floor(idx / 7)}{int(idx % 7)}{idx_str}'] = round(currlist[idx])

    valdict['id'] += 1
    return valdict


def make_argstring(values):
    argstring = ''
    for k, v in values.items():
        argstring += f'{k}_{v.item() if isinstance(v, torch.Tensor) else v}' + ' '
    return argstring


def normalize(val, mode):
    return val * 2.0 - 1.0
