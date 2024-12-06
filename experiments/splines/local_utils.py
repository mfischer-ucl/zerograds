import os.path

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_hparams(mode):
    hparams = {'k_beta': 3,  # proxy update steps
               'k_alpha': 1,  # param update steps
               'sigma': 0.125,  # normal distribution spread
               'epochs': 40000,
               'lr_beta': 1e-4,
               'lr_alpha': 1e-4,
               'device': 'cuda',
               'criterion': torch.nn.MSELoss(),

               'smooth_loss': False,
               'inner_sigma': 0.33,

               # proxy and sampling parameters
               'proxy': 'NEURAL',  # 'NEURAL', 'CVPR', 'AD', 'FD', 'QUADRATIC'
               'sampler': 'gaussian',  # 'uniform', 'gaussian', 'gradgaussian', 'mcmc', 'FD'
               'fdRadius': 0.025,  # 0.025 for r_channel example
               'quadraticproxy_a': 1.0,
               'quadraticproxy_b': 0.0,
               'quadraticproxy_c': 0.0,

               # CVPR and gradgaussian parameters
               'sigma_min': 0.01,
               'anneal_end': 500,
               'anneal_start': 100,
               'is_antithetic': False,
               'use_sigma_annealing': False,

               # problem domain for uniform sampling
               'domain_low': -2.0,
               'domain_high': 2.0,

               # network parameters
               # 'batchsize': 1,
               # 'num_layers': 3,
               # 'num_neurons': 64,
               # 'num_encodings': 1,
               # 'is_logspace': False,
               # 'use_posEnc': False,
               'batchsize': 1,
               'num_layers': 8,
               'num_neurons': 128,
               'num_encodings': 3,
               'is_logspace': False,
               'use_posEnc': True,

               # rendering parameters for mitsuba - not used when using blender
               'spp': 16,  # not used when using blender
               'res_x': 40,  # not used when using blender
               'res_y': 40,  # not used when using blender
               'max_depth': 2,  # not used when using blender
               'integrator': 'path'}  # not used when using blender
    return hparams


def get_init_from_weights(path):
    weights = torch.load(os.path.join('resources', path))['model']
    res = torch.cat([v.flatten() for k, v in weights.items()])
    return res


def set_weights(statedict, model):
    model.fc1.weight = torch.nn.Parameter(statedict['fc1.weight'], requires_grad=False)
    model.fc2.weight = torch.nn.Parameter(statedict['fc2.weight'], requires_grad=False)
    model.fc31.weight = torch.nn.Parameter(statedict['fc31.weight'], requires_grad=False)
    model.fc32.weight = torch.nn.Parameter(statedict['fc32.weight'], requires_grad=False)
    model.fc1.bias = torch.nn.Parameter(statedict['fc1.bias'], requires_grad=False)
    model.fc2.bias = torch.nn.Parameter(statedict['fc2.bias'], requires_grad=False)
    model.fc31.bias = torch.nn.Parameter(statedict['fc31.bias'], requires_grad=False)
    model.fc32.bias = torch.nn.Parameter(statedict['fc32.bias'], requires_grad=False)
    return model


def get_initial_and_gt(hparams, model, seed=0):
    device = hparams['device']

    # as per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html,
    # all layers are initialized from U(-k, k), where k = sqrt(1/in_features) per layer

    nparams = sum(p.numel() for p in model.parameters())
    f1, f2, f3, f_out = model.fc1.in_features, model.fc2.in_features, model.fc3.in_features, model.fc3.out_features
    k1 = (1.0 / f1) ** 0.5
    k2 = (1.0 / f2) ** 0.5
    k3 = (1.0 / f3) ** 0.5
    ndim = nparams

    n_p1 = f2 * f1 + f2
    n_p2 = f3 * f2 + f3
    n_p3 = f_out * f3 + f_out

    w1 = torch.rand(n_p1) * k1 * 2 - k1
    w2 = torch.rand(n_p2) * k2 * 2 - k2
    w3 = torch.rand(n_p3) * k3 * 2 - k3
    init = torch.cat([w1, w2, w3]).to(device).requires_grad_(True)
    gt = torch.rand(14).to(device)  # unused

    hparams['ndim'] = ndim
    hparams['theta'] = init
    hparams['gt_theta'] = gt


def get_defaults(mode, device, hparams, use_pretrained=False, pretr_path=None):
    pts_x = np.linspace(0, 1, num=5)
    pts_y = np.linspace(0, 1, num=5)
    pts = torch.tensor(np.stack((pts_x, pts_y)).T, dtype=torch.float32, device=device).flatten()
    defaults = {'pts': pts,
                'device': device}

    from vae import TmpMLP
    in_dim = 256 if not 'stochastic' in mode else \
        (2 if 'stochastic' in mode else ValueError(f'unknown mode {mode}'))
    model = TmpMLP(in_dim=in_dim)
    defaults['mlp'] = model.to(device)

    from torchinfo import summary
    summary(model)
    return defaults
