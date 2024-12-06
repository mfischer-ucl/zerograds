import torch


def grid(shape):
    u = torch.linspace(-1, 1, shape[1])
    v = torch.linspace(-1, 1, shape[0])
    x, z = torch.meshgrid(v, u)
    y = torch.ones_like(x)
    t = torch.stack((x, y, z))
    return t.permute(1, 2, 0)


def get_initial_and_gt(hparams, seed):

    # gt are the overfitted spline pts
    bspline_weights = torch.load(f'resources/bspline_siggraph.pt')
    gt = bspline_weights['_data'].to(hparams['device']).flatten()

    rnoise = torch.randn_like(gt) * 0.05
    init = (torch.full_like(gt, fill_value=0.5) + rnoise).requires_grad_(True)     # init is random
    ndim = gt.numel()

    hparams['ndim'] = ndim
    hparams['theta'] = init
    hparams['gt_theta'] = gt


def get_defaults(hparams):

    if 'spline' in hparams['experiment_mode']:
        from torch_cubic_spline_grids import CubicBSplineGrid2d
        N_CONTROL_POINTS = (hparams['spline_res'], hparams['spline_res'])
        bspline = CubicBSplineGrid2d(resolution=N_CONTROL_POINTS).to(hparams['device'])

        # leave spline un-initialized
        # bspline_weights = torch.load(f'resources/bspline_{hparams["image"].pt}')
        # bspline.load_state_dict(bspline_weights)

        N = hparams['photon_res']
        cgrid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, N),
                                           torch.linspace(-1, 1, N)), -1).to(hparams['device'])

        defaults = {
            'grid': cgrid,
            'iter': -1,
            'curr_x': None,
            'spline': bspline,
            'device': hparams['device'],
            'photon_res': hparams['photon_res'],
            'caustic_res': hparams['caustic_res']
        }

    return defaults
