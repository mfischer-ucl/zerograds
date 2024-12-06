import torch
import torch.nn.functional as F


# specify your path to blender here, as a string.
# in my case, this was sth like '/home/michael/software/blender-3.3.1-linux-x64/blender'
path_to_blender = None  # (set your path here)


def get_hparams_combined(exp_name, mode):

    highdim = any([x in exp_name for x in ['mlp', 'mnist', 'vae', 'texture', 'caustic', 'mesh']])
    model_dict = {
       'num_layers': 3 if highdim is False else 8,
       'num_neurons': 64 if highdim is False else 128,
       'num_encodings': 3 if 'mlp' in exp_name else 6,
       'use_posEnc': highdim
    }

    return {
            # these are normally set per-experiment
            'sigma': None,       # normal distribution spread
            'device': None,
            'epochs': None,      # number of optimization iterations
            'lr_phi': None,      # proxy learning rate
            'lr_theta': None,    # param learning rate
            'batchsize': None,   # sampling batchsize
            'smooth_loss': None,

            'id_ext': '',

            'k_phi': 3,                     # surrogate update steps
            'k_theta': 3,                   # param update steps
            'experiment_id': exp_name,
            'criterion': F.mse_loss,
            'criterion_proxy': F.mse_loss,  # possible to update proxy w/ a different loss

            'regWeight': 0.0,               # regularization weight for optional regularization term
            'inner_sigma': 0.1,             # sigma for inner normal distribution
            'is_antithetic': True,          # use antithetic sampling
            'is_regularized': False,        # use regularization term
            'is_logspace': False,           # use logspace for proxy computation

            **model_dict,

            # for texture task
            'res': 256,

            # for caustic task
            'spline_res': 32,  # num of spline control pts
            'photon_res': 512,  # the gridres that the spline will be sampled on
            'caustic_res': 512,  # the final output image res

            # dummy argument to force initialization of cuda context
            'f': torch.tensor([1.0], device='cuda' if torch.cuda.is_available() else 'cpu')
    }
