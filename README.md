# Calculating Force in FermiNet

This repository contains the codebase for the paper [Interatomic force from neural network based variational quantum Monte Carlo](https://doi.org/10.1063/5.0112344). If you use this code in your work, please [cite our paper](#citation).

## Note

This package only works with the `jax` branch of FermiNet, and make sure you store the last checkpoint when training. You also need JAX version like `0.2.28`.

> [!TIP]
> A new version is available at [bytedance/netbos](https://github.com/bytedance/netobs), where many features are added, including support for different neural networks and calculation of some other physical quantities.

## Installation

```sh
cd <project-dir>
pip install -e .
```

Note that this will not install the FermiNet package.

## Example

See `example/main.py`

## Citation

```bibtex
@article{qian_interatomic_2022,
  title = {Interatomic Force from Neural Network Based Variational Quantum {{Monte Carlo}}},
  author = {Qian, Yubing and Fu, Weizhong and Ren, Weiluo and Chen, Ji},
  year = {2022},
  month = oct,
  journal = {The Journal of Chemical Physics},
  volume = {157},
  number = {16},
  pages = {164104},
  issn = {0021-9606},
  doi = {10.1063/5.0112344},
  copyright = {All rights reserved}
}
```
