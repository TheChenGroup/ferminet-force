# Copyright 2022 Allan Chain
# Part of the code is based on FermiNet by DeepMind, same license.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from ferminet import envelopes, networks
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from optax import OptState


class RestoredParams(TypedDict):
    atoms: jnp.ndarray
    charges: jnp.ndarray
    network: networks.LogFermiNetLike
    signed_network: networks.FermiNetLike
    params: networks.ParamTree
    data: jnp.ndarray
    t_init: int
    opt_state: "OptState"
    mcmc_width: jnp.ndarray


class PartialNetwork:
    func: Any
    args: tuple
    keywords: dict[str, Any]

    def __new__(cls, func, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super().__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        # Often just need log|psi(x)|.
        return self.func(*self.args, *args, **keywords)[1]


def restore_network(cfg: "ConfigDict") -> RestoredParams:
    """Restore network from checkpoint.

    Args:
        config: The training configuration.

    Returns:
        atoms: Coordinates of atoms.
        charges: Charges of atoms.
        network: Callable with signature f(data, params) which returns the log magnitude
            of the wavefunction given parameters params and configurations data.
        params: pytree of network parameters.
        data: MCMC walker configurations.
        t: number of completed iterations.
        opt_state: optimization state.
        mcmc_width: width to use in the MCMC proposal distribution.
    """
    ckpt_filename = (
        Path(cfg.log.restore_path) / f"qmcjax_ckpt_{cfg.optim.iterations-1:06d}.npz"
    )
    with open(ckpt_filename, "rb") as f:
        ckpt_data = np.load(f, allow_pickle=True)
        data = ckpt_data["data"]
        params = ckpt_data["params"].tolist()
        if "input" not in params:
            params["input"] = {}
        t_init = ckpt_data["t"].tolist() + 1
        opt_state = ckpt_data["opt_state"].tolist()
        mcmc_width = jnp.array(ckpt_data["mcmc_width"].tolist())

    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    spins = cfg.system.electrons

    _, signed_network, _ = networks.make_fermi_net(
        atoms,
        spins,
        charges,
        envelope=envelopes.make_isotropic_envelope(),
        feature_layer=networks.make_ferminet_features(
            charges,
            cfg.system.electrons,
            cfg.system.ndim,
        ),
        bias_orbitals=cfg.network.bias_orbitals,
        use_last_layer=cfg.network.use_last_layer,
        hf_solution=None,
        full_det=cfg.network.full_det,
        **cfg.network.detnet,
    )

    network = PartialNetwork(signed_network)

    return {
        "atoms": atoms,
        "charges": charges,
        "signed_network": signed_network,
        "network": network,
        "params": params,
        "data": data,
        "t_init": t_init,
        "opt_state": opt_state,
        "mcmc_width": mcmc_width,
    }
