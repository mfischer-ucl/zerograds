import torch
import numpy as np
import matplotlib.pyplot as plt
from config import get_hparams_combined
from train_utils_modular import get_idstring, optimize_with_proxy
from local_utils import normalize, get_initial_and_gt, get_defaults


def update_values(valdict, value, mode):
    valdict['texture'] = value
    valdict['iter'] += 1
    return valdict


def render_fn(valdict, render_args):
    res = valdict['res']
    img = valdict['texture'].reshape(3, res, res).unsqueeze(0)

    return {
        'img': img,
        'regTerm': None
    }


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / image.mean() - 1.0
    scaled_ref = ref / ref.mean() - 1.0
    return torch.nn.MSELoss()(scaled_image, scaled_ref)


if __name__ == '__main__':
    experiment_id = 'texture'

    hparams = get_hparams_combined(experiment_id, None)

    defaults = get_defaults(hparams)

    hparams['k_theta'] = 1
    hparams['id_ext'] = 'texture'

    hparams['sigma'] = 0.025
    hparams['lr_phi'] = 1e-4
    hparams['batchsize'] = 20
    hparams['lr_theta'] = 1e-5
    hparams['device'] = 'cuda'
    hparams['epochs'] = 100_000
    hparams['smooth_loss'] = True
    hparams['is_antithetic'] = True

    hparams['criterion'] = scale_independent_loss
    hparams['criterion_proxy'] = scale_independent_loss

    np.random.seed(0)
    torch.manual_seed(0)

    # setup initial translation and gt translation
    get_initial_and_gt(hparams, seed=0)

    optimize_with_proxy(hparams=hparams,
                        defaults=defaults,
                        render_fn=render_fn,
                        update_fn=update_values,
                        normalize_fn=normalize,
                        plot_initial=True,
                        idstring=get_idstring(hparams, 0))
