# Calculating Force in FermiNet

This is a package to calculate force in [FermiNet](https://github.com/deepmind/ferminet)

## Note

This package only works with the `jax` branch of FermiNet, and make sure you store the last checkpoint when training. You also need JAX version like `0.2.28`.

The code was developed before my internship at ByteDance, and this repo is unlikely to update. However, many features are added to the internal version in ByteDance, including support for different neural networks and calculation of some other physical quantities. That code is still under development.

## Installation

```sh
cd <project-dir>
pip install -e .
```

Note that this will not install the FermiNet package.

## Example

See `example/main.py`

## Citation

Please consider citing the paper:

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
