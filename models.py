import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sampler(params, update_fn, render_fn, normalize_fn, defaults, gt_img):
    # utility fn that returns the sampler object

    loc = torch.tensor(0.0).to(params['device'])
    sigma = torch.tensor(params['sigma']).to(params['device'])

    # normal for parameter sampling, innerNormal for smoothingoffset sampling
    normalDistr = torch.distributions.Normal(loc=loc, scale=sigma)
    innerNormal = torch.distributions.Normal(loc=normalDistr.loc, scale=params['inner_sigma'] * normalDistr.scale)

    # make subdict for kwargs samplers, avoid passing the entire hparams
    kwargs = {
        **{k: params[k] for k in ['experiment_id', 'smooth_loss', 'ndim', 'is_regularized',
                                  'regWeight', 'is_antithetic', 'criterion', 'device']
           }, **{'gt_img': gt_img, 'defaults': defaults, 'inner_normal': innerNormal}
    }

    return GaussianSampler(normalDistr, update_fn, render_fn, normalize_fn, kwargs=kwargs)


class GaussianSampler:
    def __init__(self, normal, update_fn, render_fn, normalize_fn, kwargs):
        self.normal = normal

        # function pointers
        self.update_fn = update_fn
        self.render_fn = render_fn
        self.normalize_fn = normalize_fn

        self.device = kwargs['device']
        self.gt_img = kwargs['gt_img']
        self.defaults = kwargs['defaults']
        self.regWeight = kwargs['regWeight']
        self.criterion = kwargs['criterion']        # the actual loss function that will be used during FW pass. MSE.
        self.smooth_loss = kwargs['smooth_loss']    # use stochastic smoothing (conv. w/ blur kernel) or hard loss?
        self.innernormal = kwargs['inner_normal']   # inner normal used for sampling during smooth_loss calc
        self.regularized = kwargs['is_regularized']
        self.experiment_id = kwargs['experiment_id']
        self.is_antithetic = kwargs['is_antithetic']

    def render_batch(self, vals, render_kwargs):
        # for each sample in vals, render image and return stacked image tensor (and regTerms if regularized)
        renderings, reg_terms = [], []
        for i in range(vals.shape[0]):
            return_vals = self.render_fn(self.update_fn(self.defaults, vals[i, ...], self.experiment_id), render_kwargs)
            renderings.append(return_vals['img'])
            reg_terms.append(return_vals['regTerm'])
        renderings_stacked = torch.cat(renderings, dim=0) if len(renderings) > 1 else renderings[0]
        regTerms_stacked = torch.stack(reg_terms) if self.regularized is True else None

        return {
            'renderings': renderings_stacked,
            'regTerms': regTerms_stacked
        }

    def eval_criterion(self, renderings):
        losses = [self.criterion(x, self.gt_img.squeeze(0)) for x in renderings]
        return torch.stack(losses)

    def sample(self, current_param, kwargs, render_kwargs=None):
        # will sample from parameter space and return the samples and their associated loss values

        with torch.no_grad():

            samples = self.normal.sample([kwargs['batchsize'], kwargs['ndim']]).to(self.device)
            if self.is_antithetic:
                samples = make_antithetic_samples(samples)
            values = current_param - samples

            if self.smooth_loss:

                ti = self.innernormal.sample(values.shape)
                perturbed_values = values - ti

                if self.regularized is False:
                    renderings = self.render_batch(perturbed_values, render_kwargs)['renderings']
                else:
                    render_results = self.render_batch(perturbed_values, render_kwargs)
                    renderings, regTerms = render_results['renderings'], render_results['regTerms']

                    if self.gt_img.shape[0] > 1:
                        # ground truth image is a batch (for eg multiview mesh task), so need to rearrange renderings
                        renderings = torch.chunk(renderings, chunks=len(regTerms), dim=0)

            else:
                if self.regularized is False:
                    renderings = self.render_batch(values, render_kwargs)['renderings']
                else:
                    render_results = self.render_batch(values, render_kwargs)
                    renderings, regTerms = render_results['renderings'], render_results['regTerms']

                    if self.gt_img.shape[0] > 1:
                        # ground truth image is a batch (for eg multiview mesh task), so need to rearrange renderings
                        renderings = torch.chunk(renderings, chunks=len(regTerms), dim=0)

            if self.regularized is False:
                f_at_samples = self.eval_criterion(renderings)
            else:
                f_at_samples = torch.stack([self.criterion(x, self.gt_img) +
                                            self.regWeight * regTerms[idx] for idx, x in enumerate(renderings)])

            return {
                'samples': values,
                'f_at_samples': f_at_samples
            }


class NeuralProxy(torch.nn.Module):
    def __init__(self, argdict, normalize_fn):
        super().__init__()
        self.ctx = {}

        self.device = argdict['device']
        self.normalize_fn = normalize_fn
        self.is_logspace = argdict['is_logspace']
        self.criterion = argdict['criterion_proxy']

        self.proxy = ProxyLoss(problem_dim=argdict['ndim'], use_pe=argdict['use_posEnc'],
                               n_enc=argdict['num_encodings'],
                               out_dimension=1, n_layers=argdict['num_layers'],
                               n_hidden=argdict['num_neurons']).to(self.device)

    def forward(self, current_param, samples, f_at_samples, kwargs):
        loss = self.proxy(self.normalize_fn(current_param, kwargs['experiment_id']))
        grad = torch.autograd.grad(loss, current_param)[0]

        # save for proxy update
        self.ctx['experiment_id'] = kwargs['experiment_id']
        self.ctx['last_samples'] = samples
        self.ctx['f_at_samples'] = f_at_samples

        return grad

    def update(self, optim):
        self.ctx['last_samples'].requires_grad_(True)

        pred_losses = self.proxy(self.normalize_fn(self.ctx['last_samples'], self.ctx['experiment_id']))
        true_losses = self.ctx['f_at_samples']

        if self.is_logspace:
            true_losses = torch.log(true_losses.squeeze() + 1.0)

        meta_loss = self.criterion(pred_losses.squeeze(), true_losses.squeeze())
        optim.zero_grad()
        meta_loss.backward()
        optim.step()

    def query_lossproxy(self, parameter, expmode=None):
        if expmode is None:
            expmode = self.ctx['experiment_id']
        return self.proxy(self.normalize_fn(parameter, expmode)).squeeze()


class ProxyLoss(nn.Module):
    # the MLP that the NeuralProxy is using to learn the loss landscape.
    def __init__(self, problem_dim, use_pe, n_enc, out_dimension, n_layers, n_hidden):
        super(ProxyLoss, self).__init__()
        self.use_pe = use_pe
        self.num_enc = n_enc

        infeatures = problem_dim if use_pe is False else 2 * n_enc * problem_dim + problem_dim
        self.layers = nn.ModuleList([nn.Linear(infeatures, n_hidden)])
        for j in range(n_layers - 1):
            self.layers.append((nn.Linear(n_hidden, n_hidden)))
        self.layers.append(nn.Linear(n_hidden, out_dimension))

    def forward(self, x, kwargs=None):
        if self.use_pe is True:
            x = positional_encoding(x, num_encoding_functions=self.num_enc)

        for j in range(len(self.layers)):
            x = F.leaky_relu(self.layers[j](x))
        return x


def make_antithetic_samples(x):
    # elements are concatenated, then the first sample is negated, e.g.,
    # a = torch.tensor([
    #       [1.0, 2.0],
    #       [2.0, 3.0]])
    # becomes
    #       [-1., -2.],
    #       [ 1.,  2.],
    #       [-2., -3.],
    #       [ 2.,  3.]

    x_antith = torch.repeat_interleave(x, repeats=2, dim=0)
    x_antith[::2, :] *= (-1.0)
    return x_antith


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    # from the NeRF codebase
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
