from typing import Protocol, Tuple

from jax import numpy as jnp

ParamTree = jnp.ndarray


class FermiNetLike(Protocol):
    def __call__(
        self, params: ParamTree, electrons: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the sign and log magnitude of the wavefunction.
        Args:
          params: network parameters.
          electrons: electron positions, shape (nelectrons*ndim), where ndim is the
            dimensionality of the system.
        """


class LogFermiNetLike(Protocol):
    def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction.
        Args:
          params: network parameters.
          electrons: electron positions, shape (nelectrons*ndim), where ndim is the
            dimensionality of the system.
        """


class LogFermiNetLikeWithAtoms(Protocol):
    def __call__(
        self, params: ParamTree, electrons: jnp.ndarray, atoms: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction.

        Args:
            params: network parameters.
            electrons: electron positions, shape (nelectrons*ndim), where ndim is the
                dimensionality of the system.
            atoms: Shape (natom, ndim). Positions of the atoms.
        """
