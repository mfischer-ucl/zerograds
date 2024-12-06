import os.path
import time
import json
import torch
import numpy as np
from torchinfo import summary
import matplotlib.pyplot as plt
from general_utils import show_images
from torchvision.utils import save_image
from models import NeuralProxy, create_sampler
from tqdm import tqdm


def render_image(values, curr_theta, update_fn, render_fn, render_kwargs):
    """
    Helper function that both updates the values (in-place) and renders and returns the image
    :param values: parameter dict, containing the parameters
    :param curr_theta: the current parameter value
    :param update_fn: function pointer to a function that updates the values in the parameter dict
    :param render_fn: function that renders an image, returns [B, C, H, W]
    :param render_kwargs: additional params to pass to the render_fn
    :return: the rendered image, shape [B, C, H, W]
    """
    updated_values = update_fn(values, curr_theta, render_kwargs['experiment_id'])
    image = render_fn(updated_values, render_kwargs)['img']
    return image


def optimize_with_proxy(
        hparams,                    # hyperparameters, should be defined on the render_side
        defaults,                   # dict with default values, will be updated by update_fn
        render_fn,                  # function ptr that renders the image and returns image and optional regTerm
        update_fn,                  # function ptr that updates the current parameters
        normalize_fn,               # function ptr that takes the param and normalizes it
        plot_interval=500,          # if plot_interm is true, how often to plot
        plot_initial=False,
        plot_interm_or_final=False,
        idstring='',                # optional id-extension to disambiguate runs during saving
        render_kwargs=None,         # can be used to pass data directly to the render_fn
        writer=None,                # optinal tensorboard writer
        callback=None,              # called at end of every iteration, can be used to pass data back to render side
        provided_gt_img=None        # option to manually provide gt image
):

    print(f"Hparams:", {k: v for k, v in hparams.items() if k not in ['theta', 'gt_theta']})

    ###############################
    # setting things up, rendering gt img, plotting, ...
    ###############################

    if render_kwargs is None:
        render_kwargs = {'experiment_id': hparams['experiment_id']}

    theta, gt_theta = hparams['theta'], hparams['gt_theta']
    hparams['init_backup'] = hparams['theta'].detach().clone()  # save init backup for later rendering

    if writer is not None:
        writer.add_text('hparams', json.dumps(clean_dict(hparams)))
        writer.add_text('idstring', get_idstring(hparams, seed=0))
    if not os.path.exists('results'):
        os.makedirs('results')

    # render gt and init img
    init_img = render_image(defaults, theta, update_fn, render_fn, render_kwargs)
    if provided_gt_img is None:
        gt_img = render_image(defaults, gt_theta, update_fn, render_fn, render_kwargs)
    else:
        gt_img = provided_gt_img

    if plot_initial:
        if writer is not None:
            writer.add_images('init img', img_tensor=torch.cat([init_img, gt_img], dim=-1), global_step=0)
        else:
            show_images(init_img=init_img, ref_img=gt_img)

    # init sampler and proxy based on hparams
    sampler = create_sampler(hparams, update_fn, render_fn, normalize_fn, defaults, gt_img=gt_img)
    proxy = NeuralProxy(hparams, normalize_fn)

    # init optimizers
    optim_theta = torch.optim.Adam([theta], lr=hparams['lr_theta'])
    optim_proxy = torch.optim.Adam(proxy.parameters(), lr=hparams['lr_phi'])

    # setup kwargs: subdict that will be used during optimization
    kwargs = {k: hparams[k] for k in [
        'experiment_id', 'batchsize', 'device', 'ndim', 'is_antithetic', 'sigma', 'criterion'
    ]}
    kwargs['proxy'] = proxy             # some callback_fns might need this

    timings, img_losses = [], []

    print(f"Running {hparams['epochs']} epochs w/ batchsize {hparams['batchsize']}")

    ###############################
    # main neural proxy logic start
    ###############################

    bestloss = 1e12
    for it in tqdm(range(hparams['epochs'])):
        epoch_start = time.time()

        # update kwargs
        kwargs['current_iter'] = it

        # sample parameters and the objective at those parameters, x and f(x), respectively, i.e.,
        # f(x) = criterion(gt, render(x)) or its smooth counterpart
        sample_results = sampler.sample(current_param=theta, kwargs=kwargs)

        for k_theta in range(hparams['k_theta']):

            # get gradient for theta from the proxy
            grad = proxy(current_param=theta,
                         samples=sample_results['samples'],
                         f_at_samples=sample_results['f_at_samples'],
                         kwargs=kwargs)

            # update parameter value through gradient descent
            optim_theta.zero_grad()
            theta.grad = grad
            optim_theta.step()

        # update proxy
        for k_phi in range(hparams['k_phi']):
            proxy.update(optim_proxy)

        ###############################
        # printing, logging, saving, ...
        ###############################

        if callback is not None:
            # give the results back to function side, for the user to do whatever
            callback(kwargs, theta, gt_theta)

        # -- logging and plotting:
        timings.append(time.time() - epoch_start)

        final_img = render_image(defaults, theta, update_fn, render_fn, render_kwargs)
        img_losses.append(hparams['criterion'](final_img, gt_img).item())

        if writer is not None:
            writer.add_scalar('img_loss', img_losses[-1], global_step=it)

        if (it + 1) % 5 == 0 or it == 0:
            print(f"Iter. {it + 1}/{hparams['epochs']} - Params: {theta.tolist() if theta.numel() <= 6 else ''} - "
                  f"Img.Loss: {img_losses[-1]:.4f} - Time: {np.mean(timings[-5:]):.4f}s")

        if img_losses[-1] < bestloss:
            current_best = theta.detach().clone()
            bestloss = img_losses[-1]

        if (it + 1) % 100 == 0 and it > 0:
            # save intermediate best, will be updated once more at the end of optimization
            torch.save(current_best, f'results/log_{idstring}_bestparams.pt')

        if plot_interm_or_final and (it + 1) % plot_interval == 0:
            if writer is not None:
                writer.add_images('images', img_tensor=torch.cat([final_img, gt_img], dim=-1), global_step=it)
            else:
                show_images(final_img, gt_img, titles=[f'iter {it + 1}', 'Reference'], suptitle=idstring)
                plt.plot(np.array(img_losses) / max(img_losses), label='Img.Loss')
                plt.legend()
                plt.show()

    # plot final ...
    if plot_interm_or_final:
        plt.plot(np.array(img_losses), label='Img.Loss')
        plt.title('Final, after {} epochs'.format(hparams['epochs']))
        plt.legend()
        plt.show()
        show_images(final_img, gt_img, titles=[f'iter {it + 1}', 'Reference'], suptitle=idstring)

    # save results as json dict
    res = {'defaults': clean_dict(defaults),
           'hparams': clean_dict(hparams),
           'img_errors': [x.item() if isinstance(x, torch.Tensor) else x for x in img_losses],
           'target_value': gt_theta.tolist(),
           'final_value': theta.tolist(),
           'avg_iter_time': sum(timings) / len(timings)}

    show_images(final_img, gt_img, ['Final', 'GT'], save=True, savepath=f'results/logs_{idstring}_finalImg.png')
    with open(f'results/logs_{idstring}.json', 'w') as fp:
        json.dump(res, fp, indent=4)

    # save best from training run
    torch.save(current_best, f'results/log_{idstring}_bestparams.pt')

    print("Saved results to {}".format(f'results/logs_{idstring}.json'))
    print("Done Training.")

    return theta.detach().clone()   # return the final theta


def clean_dict(d):
    # makes a copy that removes unwanted keys and converts all potential tensors to list of floats
    # so they can be serialized to json
    bad_keys = ['criterion', 'criterion_proxy', 'params', 'scene', 'init_vpos', 'mlp']
    new_dict = {}
    for k, v in d.items():
        if k in bad_keys: continue
        if isinstance(v, torch.nn.Module): continue
        if isinstance(v, torch.Tensor):
            v = v.squeeze().item() if v.numel() <= 1 else v.tolist()
        new_dict[k] = v
    return new_dict


def get_idstring(hparams, seed):
    smooth = '' if hparams['smooth_loss'] is False else 'smooth'
    at = '_AT' if hparams['is_antithetic'] else ''
    idstring = f"{hparams['experiment_id']}_gaussian{smooth}sampler_" \
               f"_bs{hparams['batchsize']}_kphi{hparams['k_phi']}" \
               f"{at}_{hparams['id_ext']}_seed{seed}"
    return idstring


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / image.mean() - 1.0
    scaled_ref = ref / ref.mean() - 1.0
    return torch.nn.MSELoss()(scaled_image, scaled_ref)
