import torch
import numpy as np

from config import get_hparams_combined, path_to_blender
from blender_interfacing import init_blender, render_blender, close_blender
from local_utils import normalize, get_shmpath, set_initial_and_gt, make_argstring, \
    get_defaults, update_binary
from train_utils_modular import optimize_with_proxy, get_idstring


def update_values(valdict, value, mode):
    value = torch.clamp(value, 0.0, 1.0)  # bc we dont know how to handle e.g. -1.2 on blender side
    return update_binary(valdict, value)


def render_fn(values, render_args):
    img = render_blender(values, active_socket, make_argstring, shmpath, verbose=False, device=hparams['device'])
    return {
        'img': img,
        'regTerm': None
    }


if __name__ == '__main__':
    mode = 'binary_highdim'

    hparams = get_hparams_combined(mode, None)
    defaults = get_defaults(mode)

    # =====================================================

    hparams['k_theta'] = 1
    hparams['id_ext'] = 'leddisplay'

    hparams['sigma'] = 0.33
    hparams['lr_phi'] = 1e-3
    hparams['batchsize'] = 1
    hparams['lr_theta'] = 1e-3
    hparams['device'] = 'cpu'
    hparams['epochs'] = 2500
    hparams['smooth_loss'] = True
    hparams['is_antithetic'] = True

    shmpath = get_shmpath()
    blendpath = './resources/led_display_highDim.blend'

    # launch blender as non-blocking subprocess in the background, call it for rendering later
    active_socket = init_blender(path_to_blender, blendpath, 11116, shmpath, suppress_output=True, sleepsecs=3.0)

    np.random.seed(0)
    torch.manual_seed(0)
    set_initial_and_gt(hparams, seed=0)

    optimize_with_proxy(hparams=hparams,
                        defaults=defaults,
                        render_fn=render_fn,
                        update_fn=update_values,
                        normalize_fn=normalize,
                        plot_initial=True,
                        plot_interval=10000,
                        idstring=get_idstring(hparams, seed=0),
                        provided_gt_img=None)

    close_blender(active_socket)
