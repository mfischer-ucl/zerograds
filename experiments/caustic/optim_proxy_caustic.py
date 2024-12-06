import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import get_hparams_combined
from caustic_utils import create_caustic
from local_utils import get_initial_and_gt, get_defaults
from train_utils_modular import optimize_with_proxy, get_idstring


def normalize_fn(v, expmode):
    return v


def update_fn(values, param, expmode):
    values['curr_x'] = param
    return values


def render_value(values, render_args):
    with torch.no_grad():
        values['spline'].data = values['curr_x'].view(32, 32)
        img = create_caustic(bspline=values['spline'],
                             coordgrid=values['grid'],
                             photon_shape=values['photon_res'],
                             caustic_shape=values['caustic_res'],
                             device=values['device'])
    return {
        'img': img,
        'regTerm': None
    }


if __name__ == '__main__':
    mode = 'splinecaustic'
    hparams = get_hparams_combined(mode, None)
    hparams['device'] = 'cuda'
    hparams['experiment_mode'] = mode

    defaults = get_defaults(hparams)

    hparams['sigma'] = 0.0125
    hparams['batchsize'] = 10
    hparams['inner_sigma'] = 0.1
    hparams['smooth_loss'] = True
    hparams['is_antithetic'] = True

    hparams['k_phi'] = 3
    hparams['k_theta'] = 1
    hparams['lr_phi'] = 1e-4
    hparams['lr_theta'] = 1e-4

    hparams['epochs'] = 100_000
    hparams['id_ext'] = f'caustic'

    special_gt_img = torch.tensor(plt.imread(
        'resources/siggraph.png'
    )[..., :3]).mean(dim=-1)[None, None, ...].float().cuda()

    if hparams['caustic_res'] != 512:
        special_gt_img = F.interpolate(
            special_gt_img, size=(hparams['caustic_res'], hparams['caustic_res']), mode='bilinear'
        )

    plot_initial = True
    plot_interval = 200001
    plot_intermediate = False

    np.random.seed(0)
    torch.manual_seed(0)

    # setup initial translation and gt translation
    get_initial_and_gt(hparams, seed=0)

    optimize_with_proxy(hparams=hparams,
                        defaults=defaults,
                        render_fn=render_value,
                        update_fn=update_fn,
                        normalize_fn=normalize_fn,
                        plot_initial=plot_initial,
                        plot_interval=plot_interval,
                        plot_interm_or_final=plot_intermediate,
                        idstring=get_idstring(hparams, seed=0),
                        provided_gt_img=special_gt_img)
