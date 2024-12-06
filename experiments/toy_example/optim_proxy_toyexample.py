import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import glob
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt


from config import get_hparams_combined
from train_utils_modular import optimize_with_proxy


def get_initial_and_gt(hparams, seed=0):
    device = hparams['device']

    ndim = 1
    init = torch.tensor([1.1], requires_grad=True, device=device)
    gt = torch.tensor([4.], device=device)

    hparams['ndim'] = ndim
    hparams['theta'] = init
    hparams['gt_theta'] = gt


def get_defaults(hparams):
    return {'curr_x': None}


def normalize(val, mode):
    return val


def update_values(valdict, value, mode):
    valdict['curr_x'] = value
    return valdict


def render_fn(values, render_args):
    """
    Render function should always return a dict of img and regTerm, since, for some tasks, we might
    no longer have access to the regTerm / other values computed during the rendering pass later on.
    """
    curr_val = values['curr_x'].detach().cpu().numpy()
    curr_fx = spline(curr_val)[0]  # function value at the current position

    regTerm = values['curr_x'].abs().sum()  # some arbitrary regularization term
    return {
        'img': torch.full([1, 3, 2, 2], fill_value=curr_fx),
        'regTerm': regTerm
    }


def callback_fn(kwargs, curr_x, gt_x):

    plot_n = 50
    if kwargs['current_iter'] % plot_n != 0:
        return

    x_test = torch.linspace(0.0, 8.0, steps=200).reshape(200, 1)
    pred_loss = kwargs['proxy'].query_lossproxy(x_test).detach().cpu()

    # to calculate the real loss landscape, evaluate spline subtract from GT
    xval = curr_x.detach().cpu().numpy()
    yval = spline(xval)

    gt_xval = gt_x.detach().cpu().numpy()
    gt_yval = spline(gt_xval)

    if kwargs['criterion'] == torch.nn.functional.mse_loss:
        real_loss = spline(x_test.numpy()) ** 2
        plt.ylim([-25.5, 50.0])
        func_surface = spline(x_test.numpy())
        plt.plot(x_test, func_surface, c='black', label='f(x)')
    else:
        raise NotImplementedError('unknown loss')

    loss = ((curr_x - gt_x) ** 2).mean()

    n_offs, yscale = 1.5, 5.0
    normal = torch.distributions.Normal(loc=curr_x.item(), scale=kwargs['sigma'])
    pts_around_x = torch.linspace(start=curr_x.item() - n_offs, end=curr_x.item() + n_offs, steps=100)
    pts_at_normal = normal.log_prob(pts_around_x).exp() * yscale + yval

    plt.plot(x_test, real_loss, label='loss landscape')
    plt.plot(x_test, pred_loss, label='proxy')

    plt.plot(pts_around_x, pts_at_normal, c='grey', alpha=0.75, label='Sampling Gaussian')
    plt.scatter(xval, yval, c='green', marker='o', s=50, label='current')
    plt.scatter(gt_xval, gt_yval, c='red', marker='o', s=30, label='minimum')
    plt.xlabel('Parameter')
    plt.ylabel('Loss Landscape')
    plt.legend(loc='upper right')
    plt.title(f'Iter {kwargs["current_iter"]} - Loss: {loss.item():.4f}')

    # plt.show()
    plt.savefig(f'results/frames/iter{kwargs["current_iter"]}.png')
    plt.close('all')


if __name__ == '__main__':
    experiment_id = 'toyexample'

    # preparations: make spline for toy example, remove old frames
    x = np.array([-100, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 100])
    y = np.array([9, 7.5, 7.0, 6.5, 6, 4, 3.5, 0, 2, 1, 6, 8, 9])
    spline = scipy.interpolate.Akima1DInterpolator(x, y)

    framedir = './results/frames'
    os.makedirs(framedir, exist_ok=True)
    [os.remove(os.path.join(framedir, x)) for x in os.listdir(framedir)]

    hparams = get_hparams_combined(experiment_id, mode=None)
    defaults = get_defaults(hparams)

    # set hyperparameters:
    hparams['sigma'] = 0.33
    hparams['device'] = 'cpu'
    hparams['epochs'] = 6000
    hparams['lr_phi'] = 1e-3
    hparams['lr_theta'] = 5e-4
    hparams['batchsize'] = 10
    hparams['smooth_loss'] = False
    hparams['is_antithetic'] = True

    np.random.seed(42)
    torch.manual_seed(42)

    # plotting ?
    plot_interval = 100000
    plot_initial = False
    plot_intermediate = False

    # setup initial and gt_value (if known)
    get_initial_and_gt(hparams, seed=0)

    optimize_with_proxy(hparams=hparams,
                        defaults=defaults,
                        render_fn=render_fn,
                        update_fn=update_values,
                        normalize_fn=normalize,
                        render_kwargs=None,
                        plot_initial=plot_initial,
                        plot_interval=plot_interval,
                        plot_interm_or_final=plot_intermediate,
                        idstring="toyexample",
                        callback=callback_fn)

    # make video from frames, for visualization:
    frame_files = sorted(glob.glob('results/frames/iter*.png'),
                         key=lambda x: int(x.split('iter')[1].split('.png')[0]))
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, 20, (width, height))
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as output.mp4")

