import torch
import numpy as np
import matplotlib.pyplot as plt
from config import get_hparams_combined
from train_utils_modular import optimize_with_proxy, get_idstring


def get_defaults(hparams):
    return {'device': hparams['device']}


def get_initial_and_gt(hparams, seed):
    ndim = 2
    init_vals = [np.random.uniform(low=-1.5, high=2.0),
                 np.random.uniform(low=-0.5, high=3.0)]
    init = torch.tensor(init_vals, dtype=torch.float32, device=hparams['device'], requires_grad=True)
    gt = torch.tensor([1.0, 1.0], device=hparams['device'])
    hparams['ndim'] = ndim
    hparams['theta'] = init
    hparams['gt_theta'] = gt


def rosenbrock(x, y):
    a = 1.
    b = 100.
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def normalize_fn(v, expmode):
    return v / 4.  # normalize to [0, 1]


def update_fn(values, coords, expmode):
    values['coords'] = coords
    return values


def render_fn(values, render_args):
    curr_coords = values['coords']
    curr = rosenbrock(x=curr_coords[0], y=curr_coords[1])
    result = torch.full(size=(1, 3, 4, 4), fill_value=curr.item(), device=values['device'])
    return {
        'img': result,
        'regTerm': None
    }


def plot_rosenbrock():
    n_steps = 2000
    x, y = np.linspace(-1.5, 2., n_steps), np.linspace(-0.5, 3., n_steps)
    xx, yy = np.meshgrid(x, y)
    zz = rosenbrock(xx, yy)
    min_idx = np.unravel_index(zz.argmin(), zz.shape)
    print(f"Best plotted value at x={x[min_idx[0]]:.4f}/y={y[min_idx[1]]:.4f} - Achieved Cost of {zz.min()}")

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # nrows, ncols, idx
    ax.plot_surface(xx, yy, zz, cmap='jet', linewidth=0, antialiased=True)
    ax.set_title('3D Loss Landscape')

    # 2d contour plot
    ax = fig.add_subplot(1, 2, 2)
    CS = ax.contour(xx, yy, zz, levels=10)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('2D Contour Plot')

    plt.show()


if __name__ == '__main__':
    experiment_id = 'rosenbrock'

    torch.manual_seed(42)
    np.random.seed(42)

    plot_rosenbrock_fn = True

    if plot_rosenbrock_fn:
        plot_rosenbrock()

    hparams = get_hparams_combined(experiment_id, mode=None)

    hparams['sigma'] = 0.33
    hparams['device'] = 'cpu'
    hparams['epochs'] = 10000
    hparams['lr_phi'] = 1e-3
    hparams['k_theta'] = 1
    hparams['lr_theta'] = 5e-4
    hparams['batchsize'] = 4
    hparams['smooth_loss'] = True
    hparams['is_antithetic'] = True

    defaults = get_defaults(hparams)

    np.random.seed(42)
    torch.manual_seed(42)

    # setup initial translation and gt translation
    get_initial_and_gt(hparams, seed=0)

    optimize_with_proxy(hparams=hparams,
                        defaults=defaults,
                        render_fn=render_fn,
                        update_fn=update_fn,
                        normalize_fn=normalize_fn,
                        plot_initial=False,
                        plot_interval=10000,
                        plot_interm_or_final=False,
                        idstring=get_idstring(hparams, seed=0),
                        provided_gt_img=None
                        )
