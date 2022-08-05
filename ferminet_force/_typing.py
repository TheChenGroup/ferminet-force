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

from typing import NamedTuple, Tuple

import jax
from ferminet import networks
from jax import numpy as jnp
from typing_extensions import Protocol


class EnergyState(NamedTuple):
    el_all: jnp.ndarray
    el_term_all: jnp.ndarray
    ev_term_coeff_all: jnp.ndarray


InferrenceStepVal = Tuple[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, EnergyState]


class LogFermiNetLikeWithAtoms(Protocol):
    def __call__(
        self, params: networks.ParamTree, electrons: jnp.ndarray, atoms: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            atoms: Shape (natom, ndim). Positions of the atoms.
        """
