# Patch Conv: Patch convolution to avoid large GPU memory usage of Conv2D [[Blog]](https://hanlab.mit.edu/blog/patch-conv)

![patch_conv](https://github.com/mit-han-lab/patch_conv/blob/main/assets/patch_conv.jpg)

## Background

In current generative models, we usually apply convolutions over large-size activations to generate high-resolution content. However, PyTorch tends to use excessive memory for these operations, potentially leading to memory shortages even on 80GB A100 GPUs. 

As shown in the figure below, memory demands for standard PyTorch convolutions drastically increase when the input size reaches 1B parameters (channel×height×width). Notably, with a kernel size of 7×7, the 80GB A100 GPUs would trigger Out of Memory (OOM) errors. Inputs exceeding 2B parameters can further cause 3×3 convolutions exhaust all the memory and that’s just for one layer! This memory bottleneck prevents users and the community from scaling up the models to produce high-quality images.

To bypass this issue and reduce memory consumption, we propose a simple and effective solution -- Patch Conv. As shown in the above figure, similar to [SIGE](https://github.com/lmxyy/sige), Patch Conv first divides the input into several smaller patches along the height dimension while keeping some overlap between them. These patches are then reorganized into the batch dimension and fed into the original convolution to produce output patches, which are then concatenated together to form the final output. Patch Conv can reduce memory usage by over 2.4×, providing a viable workaround for the limitations of current implementations.

<p align="center">
  <img src="https://github.com/mit-han-lab/patch_conv/blob/main/assets/background.jpg" width="80%"/>
</p>

## Installation

After installing [PyTorch](https://pytorch.org), you can install `PatchConv` from PyPI:

```shell
pip install patch_conv
```

or via GitHub:

```shell
pip install git+https://github.com/mit-han-lab/patch_conv.git
```

or locally for development:

```shell
git clone git@github.com:mit-han-lab/patch_conv.git
cd patch_conv
pip install -e .
```

## Usage

All you need to do is use [`convert_model`](https://github.com/mit-han-lab/patch_conv/blob/main/patch_conv/utils.py#L6) to wrap all the `Conv2d` in your PyTorch model to our `PatchConv`. For example,

```python
from patch_conv import convert_model

model = Model(...)  # Your PyTorch model
model = convert_model(model, splits=4)  # The only modification you need to make

with torch.no_grad():
    model(...)  # Run the model in the original way
```


## Performance

![performance](https://github.com/mit-han-lab/patch_conv/blob/main/assets/performance.jpg)

Patch Conv significantly reduces memory consumption by over 2.4× across various kernel sizes and input resolutions with a marginally slower inference speed compared to vanilla convolution.

## Related Projects

* [MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning](https://arxiv.org/abs/2110.15352), Lin *et al.*, NeurIPS 2021
* [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://github.com/lmxyy/sige), Li *et al.*, NeurIPS 2022
* [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://github.com/mit-han-lab/distrifuser), Li *et al.*, CVPR 2024
