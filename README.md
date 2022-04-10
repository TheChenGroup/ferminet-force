# Calculating Force in FermiNet

This is a pacakge to calculate force in [FermiNet](https://github.com/deepmind/ferminet)

## Versioning

:warning: **Make sure to read this!**

This pacakge is developed based on `jax` branch of FermiNet. You must install the correct version.

## Installation

```sh
cd <project-dir>
pip install -e .
```

By default it will also install **my fork of FermiNet**, and this is the only tested version.

If you would like having another version of FermiNet installed, you can disable dependency auto installation by:

```sh
pip install -e . --no-deps
```

## Example

See `example/main.py`
