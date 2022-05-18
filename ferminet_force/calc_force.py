# Copyright 2022 Allan Chain

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import ferminet.constants
import ferminet.mcmc
import jax
from ferminet import hamiltonian
from jax import numpy as jnp

from ._typing import EnergyState
from .checkpoint import CheckpointManager, SimpleCheckpointManager
from .estimators import EstimatorWithEnergy
from .restore_network import restore_network

if TYPE_CHECKING:
    from ml_collections import ConfigDict


broadcast_all_local_devices = jax.pmap(lambda x: x)

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def make_different_rng_key_on_all_devices(
    rng: "jax.random.KeyArray",
) -> "jax.random.KeyArray":
    rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.random.split(rng, jax.device_count())
    return broadcast_all_local_devices(rng)


def calc_force(
    cfg: "ConfigDict",
    estimator_class,
    steps: int = 10,
    mcmc_steps: int = 10,
    mcmc_burn_in: int = 100,
    split_chunks: Optional[int] = None,
    random_seed: Optional[int] = None,
    checkpoint_mgr: Optional[CheckpointManager] = None,
    jit_loop: bool = False,
) -> dict[str, Any]:
    """Run force inference for a given molecule.

    Args:
        cfg: Bare FermiNet config object.
        create_fl_fun: Callable with signature f(network, atoms, charges) which
            returns function fl_fun with signature fl_fun(params, x)
        steps (int, optional): Steps to run inferrence. Defaults to 10.
        mcmc_steps (int, optional): Steps to run MCMC. Defaults to 10.
        mcmc_burn_in (int, optional): Burn in steps for MCMC. Defaults to 100.
        split_chunks (int, optional): Number of chunks to split the data into.
            Useful when memory is limited. Defaults to None, which means no split.
        jit_loop (bool, optional): Whether to use JIT-ed loop. Defaults to False.

    Returns:
        A dict with the following keys:
          - metadata: Metadata about the run.
            - version
            - steps
            - estimator
            - mcmc_steps
            - mcmc_burn_in
            - seed
          - force: The force results for all steps.
          * energy: The energy results for all steps.
          * pulay_term: The pulay_term, that is the term involving energy.
          * pulay_coeff: The coefficient for Ev.

        Terms starting with a star are only present if using LocalEnergySolver.
    """
    if checkpoint_mgr is None:
        checkpoint_mgr = SimpleCheckpointManager(cfg, estimator_class)

    restored_params = restore_network(cfg)
    atoms = restored_params["atoms"]
    charges = restored_params["charges"]
    network = restored_params["network"]
    params = restored_params["params"]
    data = restored_params["data"]
    mcmc_width = restored_params["mcmc_width"]

    if random_seed is None:
        random_seed = int(1e6 * time.time())
    key = jax.random.PRNGKey(random_seed)

    estimator = estimator_class(network, atoms, charges)
    solver: Solver
    if issubclass(estimator_class, EstimatorWithEnergy):
        el_fun = hamiltonian.local_energy(network, atoms, charges)
        solver = LocalEnergySolver(estimator.f_l, el_fun, split_chunks)
    else:
        solver = SimpleSolver(estimator.f_l, split_chunks)

    InferrenceStepVal = tuple[
        "jax.random.KeyArray", jnp.ndarray, jnp.ndarray, EnergyState
    ]

    def inferrence_step(i: jnp.int64, val: InferrenceStepVal) -> InferrenceStepVal:
        sharded_key, data, force_all, state = val
        sharded_key, subkeys = p_split(sharded_key)
        force_result, state = solver.local_force(i, params, data, state)
        force_all = force_all.at[i].set(force_result)
        # TODO: adapt mcmc width
        data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
        return (sharded_key, data, force_all, state)

    mcmc_step = ferminet.mcmc.make_mcmc_step(
        jax.vmap(network, (None, 0), 0),
        cfg.batch_size // jax.device_count(),
        steps=mcmc_steps,
        atoms=None,
        one_electron_moves=False,
    )
    mcmc_step = jax.pmap(
        mcmc_step, donate_argnums=1, axis_name=ferminet.constants.PMAP_AXIS_NAME
    )

    sharded_key = make_different_rng_key_on_all_devices(key)
    init_step, new_data, force_all, state = checkpoint_mgr.restore(steps)
    if new_data is not None:
        data = new_data
    init_val: InferrenceStepVal = (sharded_key, data, force_all, state)

    for _ in range(mcmc_burn_in):
        sharded_key, subkeys = p_split(sharded_key)
        data, pmove = mcmc_step(params, data, subkeys, mcmc_width)

    if jit_loop:
        sharded_key, data, force_all, state = jax.lax.fori_loop(
            init_step, steps, inferrence_step, init_val
        )
    else:
        for i in range(init_step, steps):
            sharded_key, data, force_all, state = inferrence_step(
                i, (sharded_key, data, force_all, state)
            )
            checkpoint_mgr.save(i, data, force_all, state)
        checkpoint_mgr.save(i, data, force_all, state, force_save=True)

    return {
        "metadata": {
            "version": 5,
            "steps": steps,
            "estimator": estimator_class.__name__,
            "mcmc_steps": mcmc_steps,
            "mcmc_burn_in": mcmc_burn_in,
            "seed": random_seed,
        },
        **solver.finalize_force(force_all, state),
    }


class Solver(ABC):
    def __init__(self, fl_fun):
        self.fl_fun = fl_fun

    @abstractmethod
    def local_force(
        self, i: jnp.int64, params: jnp.ndarray, data: jnp.ndarray, state: EnergyState
    ) -> tuple[jnp.ndarray, EnergyState]:
        ...

    @staticmethod
    @abstractmethod
    def finalize_force(
        force_all: jnp.ndarray, state: EnergyState
    ) -> dict[str, jnp.ndarray]:
        ...


class SimpleSolver(Solver):
    def __init__(self, fl_fun, split_chunks: Optional[int] = None):
        super().__init__(fl_fun)
        self.split_chunks = split_chunks
        self.batch_local_force = jax.pmap(
            jax.vmap(fl_fun, in_axes=(None, 0), out_axes=0)
        )

    def local_force(
        self, i: jnp.int64, params: jnp.ndarray, data: jnp.ndarray, state: EnergyState
    ) -> tuple[jnp.ndarray, EnergyState]:
        if self.split_chunks is not None:
            f_l = jnp.concatenate(
                [
                    self.batch_local_force(params, data_chunk)
                    for data_chunk in jnp.split(data, self.split_chunks, axis=1)
                ]
            )

        else:
            f_l = self.batch_local_force(params, data)
        force_result = jnp.mean(f_l, axis=(0, 1))
        return force_result, state

    @staticmethod
    def finalize_force(
        force_all: jnp.ndarray, state: EnergyState
    ) -> dict[str, jnp.ndarray]:
        return {"force": force_all}


class LocalEnergySolver(Solver):
    def __init__(self, fl_fun, el_fun, split_chunks: Optional[int] = None):
        super().__init__(fl_fun)
        self.split_chunks = split_chunks
        self.batch_local_force = jax.pmap(
            jax.vmap(fl_fun, in_axes=(None, 0, 0), out_axes=0)
        )
        self.batch_local_energy = jax.pmap(
            jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)
        )

    def local_force(
        self, i: jnp.int64, params: jnp.ndarray, data: jnp.ndarray, state: EnergyState
    ) -> tuple[jnp.ndarray, EnergyState]:
        e_l = self.batch_local_energy(params, data)
        if self.split_chunks is not None:
            hf_term, el_term, ev_term_coeff = [
                jnp.concatenate(x)
                for x in zip(
                    *[
                        self.batch_local_force(params, data_chunk, el_chunk)
                        for (data_chunk, el_chunk) in zip(
                            jnp.split(data, self.split_chunks, axis=1),
                            jnp.split(e_l, self.split_chunks, axis=1),
                        )
                    ]
                )
            ]
        else:
            hf_term, el_term, ev_term_coeff = self.batch_local_force(params, data, e_l)
        state = EnergyState(
            el_all=state.el_all.at[i].set(jnp.mean(e_l, axis=(0, 1))),
            el_term_all=state.el_term_all.at[i].set(jnp.mean(el_term, axis=(0, 1))),
            ev_term_coeff_all=state.ev_term_coeff_all.at[i].set(
                jnp.mean(ev_term_coeff, axis=(0, 1))
            ),
        )
        return jnp.mean(hf_term, axis=(0, 1)), state

    @staticmethod
    def finalize_force(
        force_all: jnp.ndarray, state: EnergyState
    ) -> dict[str, jnp.ndarray]:
        energy_mean = jnp.mean(state.el_all, axis=0)
        # Don't use mean of ev_term_coeff_all, because it will increase fluctuations
        product = state.ev_term_coeff_all * energy_mean
        return {
            # force_all is actually hf_term_all
            # the name is just to keep function signature the same
            "force": force_all + state.el_term_all + product,
            # hf_term can be calculated by `force - pulay_term`
            "pulay_term": state.el_term_all + product,
            "energy": state.el_all,
            "pulay_coeff": state.ev_term_coeff_all,
        }
