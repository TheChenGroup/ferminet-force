import time
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from zipfile import BadZipFile

from absl import logging
from jax import numpy as jnp

from ._typing import EnergyState

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class CheckpointManager(ABC):
    def __init__(self, config: "ConfigDict", estimator_class) -> None:
        """Create a checkpoint manager.

        Args:
            config (ConfigDict): The FermiNet config
            estimator_class (Any): The estimator class
        """
        self.config = config
        self.estimation_class = estimator_class

    @abstractmethod
    def restore(
        self, steps: int
    ) -> tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]:
        """Restore a force checkpoint

        Args:
            steps (int): Steps about to run

        Returns:
            tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]:
              - Starting iteration
              - MCMC configuration
              - Force result at each steps
              - Energy state
                  - el_all: jnp.ndarray
                  - el_term_all: jnp.ndarray
                  - ev_term_coeff_all: jnp.ndarray
        """

    @abstractmethod
    def save(
        self,
        i: int,
        data: jnp.ndarray,
        force_all: jnp.ndarray,
        state: EnergyState,
        force_save: bool = False,
    ) -> None:
        """Save a force checkpoint, optionally.

        Args:
            i (int): The current iteration
            data (jnp.ndarray): MCMC configuration
            force_all (jnp.ndarray): Force result at each steps
            state (EnergyState): Energy state
            force_save (bool): Force do the save.
                Usually used when saving the last checkpoint.
        """

    @staticmethod
    def create_empty_state(
        steps: int, atom_num: int
    ) -> tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]:
        """Creates an empty state for restore.

        Typically used when there is no checkpoint to restore.

        Args:
            steps (int): Total steps to run
            atom_num (int): Number of atoms

        Returns:
            tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]: Empty state
        """
        force_all = jnp.zeros((steps, atom_num, 3))

        state = EnergyState(
            el_all=jnp.zeros(steps),
            el_term_all=jnp.zeros((steps, atom_num, 3)),
            ev_term_coeff_all=jnp.zeros((steps, atom_num, 3)),
        )
        return 0, None, force_all, state


class SimpleCheckpointManager(CheckpointManager):
    def restore(
        self, steps: int
    ) -> tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]:
        return CheckpointManager.create_empty_state(
            steps, len(self.config.system.molecule)
        )

    def save(
        self,
        i: int,
        data: jnp.ndarray,
        force_all: jnp.ndarray,
        state: EnergyState,
        force_save: bool = False,
    ) -> None:
        logging.log_every_n_seconds(logging.INFO, "Loop %s", 5, i)


class SavingCheckpointManager(CheckpointManager):
    def __init__(
        self,
        config: "ConfigDict",
        estimator_class,
        save_interval: int = 600,
        restore_path: Optional[PathLike] = None,
        save_path: Optional[PathLike] = None,
    ) -> None:
        """Create a checkpoint manager that actually saves something.

        Args:
            config (ConfigDict): FermiNet config
            save_interval (int, optional): Minimum interval between saves in seconds.
                Defaults to 600.
            restore_path (Optional[str], optional): Path to restore from.
                Defaults to None.
            save_path (Optional[str], optional): Path to save checkpoints to.
                Defaults to None.
        """
        super().__init__(config, estimator_class)
        self.save_interval = save_interval
        self.previous_ckpt_time = time.time()

        dir_name = f"{self.estimation_class.__name__}.ckpt.d"
        if restore_path is not None:
            self.restore_path = Path(restore_path)
        else:
            self.restore_path = Path(config.log.restore_path) / dir_name
        if save_path is not None:
            self.save_path = Path(save_path)
        else:
            self.save_path = Path(config.log.save_path) / dir_name

        self.save_path.mkdir(exist_ok=True)

    def restore(
        self, steps: int
    ) -> tuple[int, Optional[jnp.ndarray], jnp.ndarray, EnergyState]:
        atom_num = len(self.config.system.molecule)
        if not self.restore_path.exists():
            return CheckpointManager.create_empty_state(steps, atom_num)

        ckpt_files = sorted(list(self.restore_path.glob("force_*.npz")), reverse=True)
        for ckpt_file in ckpt_files:
            try:
                ckpt_content = jnp.load(ckpt_file)
                old_steps_to_run = len(ckpt_content["force_all"])
                old_steps_have_run = jnp.sum(ckpt_content["force_all"][:, 0, 0] != 0)
                print(type(ckpt_content["force_all"]))
                assert (
                    old_steps_have_run < steps
                ), "We have already done that calculation!"
                if old_steps_to_run != steps:
                    _, _, force_all, old_state = CheckpointManager.create_empty_state(
                        steps, atom_num
                    )
                    steps_to_set = min(steps, old_steps_have_run)
                    print(ckpt_content["force_all"].shape)
                    force_all = force_all.at[:steps_to_set].set(
                        ckpt_content["force_all"][:steps_to_set]
                    )
                    print(force_all.shape)
                    state = EnergyState(
                        old_state.el_all.at[:steps_to_set].set(
                            ckpt_content["el_all"][:steps_to_set]
                        ),
                        old_state.el_term_all.at[:steps_to_set].set(
                            ckpt_content["el_term_all"][:steps_to_set]
                        ),
                        old_state.ev_term_coeff_all.at[:steps_to_set].set(
                            ckpt_content["ev_term_coeff_all"][:steps_to_set]
                        ),
                    )
                else:
                    force_all = ckpt_content["force_all"]
                    state = EnergyState(
                        ckpt_content["el_all"],
                        ckpt_content["el_term_all"],
                        ckpt_content["ev_term_coeff_all"],
                    )
                return ckpt_content["i"], ckpt_content["data"], force_all, state
            except (OSError, EOFError, BadZipFile):
                logging.info("Error loading %s. Trying next checkpoint...", ckpt_file)

        return CheckpointManager.create_empty_state(steps, atom_num)

    def save(
        self,
        i: int,
        data: jnp.ndarray,
        force_all: jnp.ndarray,
        state: EnergyState,
        force_save: bool = False,
    ) -> None:
        if not force_save:
            current_time = time.time()
            if current_time - self.previous_ckpt_time < self.save_interval:
                return
            self.previous_ckpt_time = current_time

        jnp.savez(
            self.save_path / f"force_{i:06}.npz",
            i=i,
            data=data,
            force_all=force_all,
            **state._asdict(),
        )
