from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from ferminet import hamiltonian, networks

from ._typing import LogFermiNetLike, LogFermiNetLikeWithAtoms, ParamTree
from .restore_network import PartialNetwork


def primitive_f_aa(atoms, charges):
    """Calculates the primitive force on the atoms.

    Args:
        atoms: Shape (natom, ndim). Positions of the atoms.
        charges: Shape (natom). Nuclear charges of the atoms.

    Returns:
        f_aa: Shape (natom, ndim). Force on the nuclei because of other nuclei.
    """
    # aa[0, 1] = x_a1 - x_a0
    aa = atoms[None, ...] - atoms[:, None]
    # || (1, natom, ndim) - (natom, 1, ndim) || = (natom, natom)
    # POSSIBLE PITFALL: norm of zero has undefined grad, as stated in FermiNet
    r_aa = jnp.linalg.norm(aa, axis=-1)
    # f_aa_matrix[0, 1] points from atom 0 to atom 1, so its force on atom 1
    # charges: (natom); aa: (natom, natom, 3); r_aa: (natom, natom, 1)
    f_aa_matrix = jnp.nan_to_num(
        (charges[None, ..., None] * charges[..., None, None])
        * aa
        / r_aa[..., None] ** 3
    )
    return jnp.sum(f_aa_matrix, axis=0)


def primitive_f_ae(ae, r_ae, charges):
    """Calculates the primitive force between the atoms and electrons.

    Args:
        ae: Shape (nelectron, natom, ndim). Relative positions of the electrons.
        r_ae: Shape (nelectron, natom, 1). Relative distances of the electrons.
            r_ae[i, j] gives the distance between electron i and atom j.
        charges: Shape (natom). Nuclear charges of the atoms.

    Returns:
        f_ae: Shape (natom, ndim). Force on the nuclei because of the electrons.
    """
    # ae: x_e - x_a
    return jnp.sum(charges[..., None] * ae / r_ae**3, axis=0)


def primitive_force(ae, r_ae, atoms, charges):
    """Calculates the primitive force on nuclei.

    Args:
        ae: Shape (nelectron, natom, ndim). Relative positions of the electrons.
        r_ae: Shape (nelectron, natom, 1). Relative distances of the electrons.
            r_ae[i, j] gives the distance between electron i and atom j.
        atoms: Shape (natom, ndim). Positions of the atoms.
        charges: Shape (natom). Nuclear charges of the atoms.

    Returns:
        f: Shape (natom, ndim). Force on the nuclei.
    """
    f_aa = primitive_f_aa(atoms, charges)
    f_ae = primitive_f_ae(ae, r_ae, charges)
    return f_aa + f_ae


class LocalForceEstimatorBase(ABC):
    """The base class for local force estimators."""

    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        """creates function to evaluate the local force.

        Args:
            f: Callable with signature f(data, params) which returns the log magnitude
                of the wavefunction given parameters params and configurations data.
            atoms: Shape (natom, ndim). Positions of the atoms.
            charges: Shape (natom). Nuclear charges of the atoms.
        """
        self.f = f
        self.atoms = atoms
        self.charges = charges


class EstimatorWithoutEnergy(ABC):
    @abstractmethod
    def f_l(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Calculates the total force.

        Args:
            params: network parameters.
            x: MCMC configuration.

        Returns:
            force: Shape (natom, ndim). Force on each atom.
        """


class EstimatorWithEnergy(ABC):
    @abstractmethod
    def f_l(
        self, params: jnp.ndarray, x: jnp.ndarray, e_l: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Calculates the total force.

        Args:
            params: network parameters.
            x: MCMC configuration.
            e_l: Local energy

        Returns:
            zv_term: Shape (natom, ndim). First 2 terms, aka ZV term.
            el_term: Shape (natom, ndim). E_L term.
            ev_term_coeff: Shape (natom, ndim). coefficient of E_v term.
        """


class PrimitiveEstimator(LocalForceEstimatorBase, EstimatorWithoutEnergy):
    def f_l(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        ae, _, r_ae, _ = networks.construct_input_features(x, self.atoms)
        return primitive_force(ae, r_ae, self.atoms, self.charges)


class PsiMinEstimatorBase(LocalForceEstimatorBase):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)

        def Q(x):
            ae, _, r_ae, _ = networks.construct_input_features(x, atoms)
            # WHY negative? Direction is not the same as the paper
            return jnp.sum(-charges[..., None] * ae / r_ae, axis=0)

        self.Q = Q


class PsiMinZVEstimator(PsiMinEstimatorBase, EstimatorWithoutEnergy):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)
        self.grad_Q = jax.jacfwd(self.Q)
        self.grad_f = jax.grad(f, argnums=1)

    def f_l(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        f_aa = primitive_f_aa(self.atoms, self.charges)
        dot_term = jnp.dot(self.grad_Q(x), self.grad_f(params, x))
        return f_aa - dot_term


class PsiMinZVZBEstimator(PsiMinEstimatorBase, EstimatorWithEnergy):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)
        self.zv_estimator = PsiMinZVEstimator(f, atoms, charges)

    def f_l(
        self, params: jnp.ndarray, x: jnp.ndarray, e_l: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        f_zv = self.zv_estimator.f_l(params, x)
        Qx = self.Q(x)
        return f_zv, 2 * e_l * Qx, -2 * Qx


class PsiDerivEstimatorBase(LocalForceEstimatorBase):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)

        f_with_atoms = PartialNetwork(f)
        del f_with_atoms.keywords["atoms"]
        # Remind f returns the log magnitude of the wavefunction
        grad_f_atoms = jax.grad(f_with_atoms, argnums=2)
        # WHY negative? Direction is not the same as the paper
        # Note \frac{\partial}{\partial \lambda} = - \frac{\partial}{\partial R}
        self.f_deriv = lambda params, x: -grad_f_atoms(params, x, atoms)


class PsiMinDerivEstimator(PsiDerivEstimatorBase, EstimatorWithEnergy):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)
        self.zv_estimator = PsiMinZVEstimator(f, atoms, charges)

    def f_l(
        self, params: jnp.ndarray, x: jnp.ndarray, e_l: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        f_zv = self.zv_estimator.f_l(params, x)
        f_deriv_val = self.f_deriv(params, x)
        return f_zv, 2 * e_l * f_deriv_val, -2 * f_deriv_val


def extended_local_kinetic_energy(f, shape):
    def _lapl_over_f(params, x):
        n = x.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.jacfwd(f, argnums=1)
        grad_f_closure = partial(grad_f, params)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[..., i] ** 2 + tangent[..., i]

        return -0.5 * jax.lax.fori_loop(0, n, _body_fun, jnp.zeros(shape))

    return _lapl_over_f


class PsiDerivEstimator(PsiDerivEstimatorBase, EstimatorWithEnergy):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)
        self.deriv_ke = extended_local_kinetic_energy(
            # log(\psi')
            lambda params, x: jnp.log(jnp.abs(self.f_deriv(params, x))) + f(params, x),
            (len(atoms), 3),
        )

    def f_l(
        self, params: jnp.ndarray, x: jnp.ndarray, e_l: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ae, _, r_ae, r_ee = networks.construct_input_features(x, self.atoms)
        deriv_potential = hamiltonian.potential_energy(
            r_ae, r_ee, self.atoms, self.charges
        )
        deriv_e_l = deriv_potential + self.deriv_ke(params, x)
        deriv_val = self.f_deriv(params, x)
        return (
            primitive_force(ae, r_ae, self.atoms, self.charges)
            + (deriv_e_l - e_l) * deriv_val,
            2 * e_l * deriv_val,
            -2 * deriv_val,
        )


def local_kinetic_energy(f: LogFermiNetLikeWithAtoms):
    """Basically the same as the original one, but exposed atoms"""

    def _lapl_over_f(params, x, atoms):
        n = x.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f, argnums=1)

        def grad_f_closure(y):
            return grad_f(params, y, atoms)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[i] ** 2 + tangent[i]

        return -0.5 * jax.lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


def potential_energy(r_ae, r_ee, atoms, charges):
    """Basically the same as the original one, but fixed r_aa for grad"""
    v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
    v_ae = -jnp.sum(charges / r_ae[..., 0])
    # Diagonals are not used, so there is no need to put them to zero
    r_aa = jnp.linalg.norm(
        atoms[None, ...] - atoms[:, None] + jnp.eye(atoms.shape[0])[..., None], axis=-1
    )
    v_aa = jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa


def local_energy_atom_exposed(f: LogFermiNetLikeWithAtoms, charges: jnp.ndarray):
    """Creates the function to evaluate the local energy, with `atoms` exposed.

    Also removed some unused args.
    """
    ke = local_kinetic_energy(f)

    def _e_l(params: ParamTree, data: jnp.ndarray, atoms: jnp.ndarray) -> jnp.ndarray:
        """Basically the same as the original one, but fixed r_ee for grad"""
        _, _, r_ae, _ = networks.construct_input_features(data, atoms)
        ee = jnp.reshape(data, (1, -1, 3)) - jnp.reshape(data, (-1, 1, 3))
        # Diagonals are not used, so there is no need to put them to zero
        r_ee = jnp.linalg.norm(
            ee + jnp.eye(ee.shape[0])[..., None], axis=-1, keepdims=True
        )

        potential = potential_energy(r_ae, r_ee, atoms, charges)
        kinetic = ke(params, data, atoms)
        return potential + kinetic

    return _e_l


class LocalEnergyDerivBase(LocalForceEstimatorBase):
    def __init__(self, f: LogFermiNetLike, atoms: jnp.ndarray, charges: jnp.ndarray):
        super().__init__(f, atoms, charges)

        grad_f_elec = jax.grad(f, argnums=1)
        self.f_deriv_elec = lambda params, x: jnp.reshape(
            grad_f_elec(params, x), (-1, 1, 3)
        )
        f_with_atoms = PartialNetwork(f)
        del f_with_atoms.keywords["atoms"]
        # Remind f returns the log magnitude of the wavefunction
        grad_f_atoms = jax.grad(f_with_atoms, argnums=2)
        # Why positive again? Sorella and Capriotti use the opposite sign as Assaraf
        self.f_deriv_atom = lambda params, x: grad_f_atoms(params, x, atoms)

        el_fun = local_energy_atom_exposed(f_with_atoms, charges)
        grad_el_atoms = jax.grad(el_fun, argnums=2)
        self.el_deriv_atom = lambda params, x: grad_el_atoms(params, x, atoms)
        grad_el_elec = jax.grad(el_fun, argnums=1)
        # Remind x is in shape (nelectrons*ndim,)
        self.el_deriv_elec = lambda params, x: jnp.reshape(
            grad_el_elec(params, x, atoms), (-1, 1, 3)
        )


class SWCTEstimator(LocalEnergyDerivBase, EstimatorWithEnergy):
    """Space warp coordinate transformation

    Eq 14 of S. Sorella and L. Capriotti, J. Chem. Phys. 133, 234111 (2010).
    """

    def f_l(
        self, params: jnp.ndarray, x: jnp.ndarray, e_l: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        omega_mat = self.omega(x)
        omega_grad = self.omega_jacfwd(jnp.reshape(x, (-1, 3)))
        zv_term = -(
            self.el_deriv_atom(params, x)
            + jnp.sum(omega_mat * self.el_deriv_elec(params, x), axis=0)
        )
        ev_term_coeff = 2 * (
            self.f_deriv_atom(params, x)
            + jnp.sum(omega_mat * self.f_deriv_elec(params, x) + omega_grad / 2, axis=0)
        )
        return zv_term, -e_l * ev_term_coeff, ev_term_coeff

    def omega(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the \\omega matrix

        Args:
            x: electron positions. Shape (nelectrons*ndim,).

        Returns:
            \\omega matrix, shape (nelectron, natom, 1)
        """
        _, _, r_ae, _ = networks.construct_input_features(x, self.atoms)
        # Remind r_ae is in shape (nelectron, natom, 1)
        f_mat = self.decay_function(r_ae)
        return f_mat / f_mat.sum(axis=1, keepdims=True)

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jacfwd, argnums=1)
    def omega_jacfwd(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the derivative for \\omega matrix by electron postion

        Args:
            x: single electron position. Shape (nelctron, ndim).
                Shape for undecorated: (ndim,)

        Returns:
            Derivative of \\omega matrix. Shape (nelctron, natom, ndim).
                Shape for undecorated: (natom,)
        """
        _, _, r_ae, _ = networks.construct_input_features(x, self.atoms)
        f_mat = self.decay_function(r_ae[1, :, 1])
        return f_mat / jnp.sum(f_mat)

    @staticmethod
    def decay_function(r_ae: jnp.ndarray) -> jnp.ndarray:
        """The fast decaying function, aka F"""
        return 1 / r_ae**4


all_estimators = {
    "prim": PrimitiveEstimator,
    "zv": PsiMinZVEstimator,
    "zvzb": PsiMinZVZBEstimator,
    "min_deriv": PsiMinDerivEstimator,
    "deriv": PsiDerivEstimator,
    "swct": SWCTEstimator,
}
