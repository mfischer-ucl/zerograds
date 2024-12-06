import os
import math
import numpy as np
import torch
import torchvision.io
import torch.nn.functional as F
from train_utils_modular import scale_independent_loss


def get_initial_and_gt(hparams, seed=0):
    device = hparams['device']
    res = hparams['res']
    ndim = res * res * 3
    refTexture = torchvision.io.read_image('resources/starrynight-512.jpg').unsqueeze(0) / 255.
    refTexture = F.interpolate(refTexture, size=res, mode='bilinear')

    gt = refTexture.clone().detach().flatten().to(device)
    const = torch.full(size=(3, res, res), fill_value=0.5, device=device).flatten()
    init = torch.tensor(const + torch.normal(0.0, 0.05, size=const.shape, device=device), requires_grad=True)

    hparams['gt_theta'] = gt
    hparams['theta'] = init
    hparams['ndim'] = ndim
    hparams['res'] = res


def get_defaults(hparams):
    res = hparams['res']
    defaults = {'texture': torch.zeros(res * res * 3),
                'res': res,
                'iter': -1}
    return defaults


def normalize(val, mode):
    return val
