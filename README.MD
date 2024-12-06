## ZeroGrads: Learning Local Surrogate Losses for Non-Differentiable Graphics

This is the offical code repository for our paper

**ZeroGrads: Learning Local Surrogate Losses for Non-Differentiable Graphics [SIGGRAPH 2024, Journal Track]**

by [Michael Fischer](https://mfischer-ucl.github.io) and [Tobias Ritschel](https://www.homepages.ucl.ac.uk/~ucactri). 
For more information, make sure to check out the [paper](https://dl.acm.org/doi/10.1145/3658173) 
and [project page](https://mfischer-ucl.github.io/zerograds/).

___
### Installation

Clone the repository, then create a new conda/mamba environment, activate it and install the dependencies.  
```
git clone https://github.com/mfischer-ucl/zerograds 
cd zerograds 
conda create -n zerograds python=3.9 
conda activate zerograds
pip install -r requirements.txt 
```

You should then be able to run the example in `examples/optimize_proxy_toyexample.py`. To do so, cd into `experiments/toy_example` and run
`export PYTHONPATH=../.. && python optimize_proxy_toyexample.py`. If all goes well, the script will run a small toy example 
optimization and output a video called `output.mp4`: 

<p align="center">
  <img src="/assets/toyexample.gif" alt="Alt Text" width="50%">
</p>

Tested with Python 3.9, PyTorch 2.0.1 and CUDA 11.7 on Ubuntu 20.04.4 x64 and an NVIDIA RTX3000 series GPU.   
___
### Usage 

ZeroGrads can optimize any black-box function by simply using the `optimize_with_proxy` routine, which doesn't
make any assumptions about the underlying renderer. All that is required is a forward function (called `render_fn` - see below)
that takes a parameter and returns a rendered image.

On a high level, ZeroGrads works by sampling the optimization space (see `models/GaussianSampler`) and using a local surrogate model (an MLP, see `models/NeuralProxy` and `models/ProxyLoss`) to approximate the
loss landscape. We can then query the gradients of this neural surrogate and use them to optimize our black-box renderer.
All of this is implemented in the `optimize_with_proxy` function. 
To get started, take a look at the `examples/`, where we provide many of the paper's experiments and results.

### Running your own experiments

To run your own experiments, you will need to slighly adapt four key functions: 

1. `render_fn`- the funtion that calls the renderer and renders an image. Its inputs are the parameter dictionary (see step 4)
and an additional `render_kwargs` dictionary that can be used to pass additional arguments to the renderer. It returns
the rendered image (PyTorch convention, i.e., `[B, C, H, W]`) and an optional regularization term that can be used 
to regularize the loss that will be computed from this image.
Note: even if the function's output is scalar, render_fn should always return an image-shaped tensor. This can easily be 
accomplished via `torch.full([1, 1, 2, 2], fill_value=scalar)`.

2. `get_initial_and_gt`- this function is responsible for setting up the optimization, i.e., to define the initial parameters, 
dimensionality, etc. If the ground truth parameters are known, they can be set here and will later be used to render the reference. Otherwise, one can set them to
zero and instead provide a reference image via the args of the `optimize_with_proxy` function. 
3. `update_fn`- A small helper function that updates the parameter dict that is passed to the `render_fn`. 
4. `get_defaults`- Returns the default parameter dictionary that is passed to the `render_fn`. Should always contain 
at least the optimization parameter and `iter`, can be extended to pass other useful things.  

Additionally, you will need to define a set of hyperparameters for your experiment, such as the learning rate and the 
number of function evaluations and epochs. You can check the examples for good starting values. 

#### A few caveats
- The first layer of the neural proxy has the dimensionality of the input space, which can be memory-intensive for high-dimensional
problems (e.g., the texture example requires 10GB of GPU memory).
- The optimization is sensitive to the choice of hyperparameters, especially the learning rate and sigma. For higher-dimensional 
experiments, we found `sigma=0.025` to be a good starting point. For the learning rates, we found `1e-4, 1e-5` to work well.
- The Blender example `examples/led_display` requires a runnable Blender installation and assumes the OS to be Linux, so we can create a ramdisk at `dev/shm`. 
You will need to adapt the paths in `config.py` example to your own Blender installation and/or system ramdisk.
The default in this repo has been set to `./experiments/led_display/results/shm_dummy` for security, but for max render speed, one can use `dev/shm` during development.
- Higher-dimensional experiments require a larger number of function evaluations to smooth out the noise in the proxy's gradient estimate. 
This will lead to longer runtimes (the texture examples takes 2hrs on my machine, for instance).
 

If you are interested in researching and improving these limitations, please don't hesitate to reach out to us.

___
[//]: # (### Disclaimer)

[//]: # ()
[//]: # (**WARNING:** This code interacts with sensitive system components, including reading from and writing to the /dev partition on Unix-based systems. Use of this code may lead to system instability, data loss, or other unintended consequences. While extensive testing has been performed without incident, we do not guarantee the code’s compatibility with all system configurations.)

[//]: # ()
[//]: # (We strongly recommend that users configure the dev/shm path to a directory under /home for safer usage.)

[//]: # ()
[//]: # (By using this code, you acknowledge that you understand the potential risks and assume full responsibility for any damage, loss, or issues caused. The code is provided "as is" without any warranties, express or implied, including but not limited to fitness for a particular purpose or non-infringement. The authors shall not be held liable for any claims, damages, or other liabilities arising from the use or misuse of this software.)

[//]: # ()
[//]: # (&#40;fyi: nothing has ever happened, we think it's safe, but we have to say this for legal reasons&#41;)

[//]: # (___)

### License 
This code is licensed under the MIT license. 
___
### Citation
If you find our work useful or plan to (re-) use parts of it in your own projects, please include the following citation:
```
@article{fischer2024zerograds,
  title={ZeroGrads: Learning Local Surrogates for Non-Differentiable Graphics},
  author={Fischer, Michael and Ritschel, Tobias},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={4},
  pages={1--15},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
