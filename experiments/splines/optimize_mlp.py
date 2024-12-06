import torch
import numpy as np
import torchvision
from torchvision import transforms

from config import get_hparams_combined
from local_utils import get_initial_and_gt, get_defaults
from train_utils_modular import optimize_with_proxy, get_idstring


def update_values(valdict, value, mode):
    # Update MLP weights with values from the input tensor
    # must reshape accordingly, since tensor is flattened
    model = valdict['mlp']
    layers = [model.fc1, model.fc2, model.fc3]
    offset = 0

    for layer in layers:
        in_features, out_features = layer.in_features, layer.out_features
        weight_size = out_features * in_features
        bias_size = out_features

        # Update the weights and bias for the current layer
        layer.weight = torch.nn.Parameter(value[offset:offset + weight_size].reshape(out_features, in_features))
        offset += weight_size
        layer.bias = torch.nn.Parameter(value[offset:offset + bias_size])
        offset += bias_size

    return valdict


def render_fn(values, render_args):
    # get input image, flatten, put through MLP, get MLP

    mlp_input = values['mlp_in']
    mlp_output = values['mlp'](mlp_input)
    out = mlp_output.reshape(28, 28).cuda()
    image = torch.cat([out.unsqueeze(0).unsqueeze(0)] * 3, dim=1)

    return {
        'img': image,
        'regTerm': None
    }


def normalize(v, expmode):
    # values already in suitable range
    return v


if __name__ == '__main__':
    mode = 'mnist_mlp'

    hparams = get_hparams_combined(mode, None)

    hparams['sigma'] = 0.025
    hparams['inner_sigma'] = 0.1
    hparams['num_encodings'] = 3

    hparams['sigma'] = 0.025
    hparams['k_phi'] = 10
    hparams['lr_phi'] = 1e-4
    hparams['k_theta'] = 4
    hparams['lr_theta'] = 1.25e-05
    hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['epochs'] = 5000
    hparams['smooth_loss'] = True
    hparams['is_antithetic'] = True
    hparams['batchsize'] = 10
    hparams['id_ext'] = 'mnistmlp'

    defaults = get_defaults(mode, hparams['device'], hparams)

    np.random.seed(42)
    torch.manual_seed(42)
    a = torch.tensor([1.0], device=hparams['device'])  # init cuda context
    update_fn = update_values

    # setup initial translation and gt translation
    get_initial_and_gt(hparams, model=defaults['mlp'], seed=7)

    train_dataset = torchvision.datasets.MNIST('resources/dataset', train=True, download=True)
    train_dataset.transform = transforms.Compose([transforms.ToTensor()])
    img = train_dataset[7][0].unsqueeze(0).cuda()
    special_gt_img = torch.cat([img] * 3, dim=1)
    defaults['mlp_in'] = torch.rand(256, device=hparams['device'])

    optimize_with_proxy(
        hparams=hparams,
        defaults=defaults,
        render_fn=render_fn,
        update_fn=update_values,
        normalize_fn=normalize,
        plot_initial=True,
        idstring=get_idstring(hparams, 7),
        provided_gt_img=special_gt_img
    )
