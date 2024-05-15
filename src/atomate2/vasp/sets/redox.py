"""Module defining VASP input set generators for defect calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.vasp.inputs import Kpoints, KpointsSupportedModes

from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun

@dataclass
class SlabRelaxSetGenerator(VaspInputGenerator):
    """Generator for atomic-only relaxation for slab calculations.
    """

    user_kpoints_settings: dict | Kpoints = field(default_factory=dict)

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "IBRION": 2,
            "ISIF": 0,
            "EDIFF": 1e-5,
            "EDIFFG": -0.02,
            "LREAL": 'auto',
            "NSW": 30,
            "ENCUT": 550,
            "LAECHG": False,
            "NELMIN": 5,
            "NELM": 30, 
            "LCHARG": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LWAVE": True,
            "KPAR": 2,
            "GGA": "PE",
            "ALGO": "Normal",
        }


@dataclass
class SlabStaticSetGenerator(VaspInputGenerator):
    """Generator for static supercell calculations.
    """

    user_kpoints_settings: dict | Kpoints = field(default_factory=dict)

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """Get updates to the INCAR for a relaxation job.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        vasprun
            A vasprun from a previous calculation.
        bandgap
            The band gap.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        return {
            "EDIFF": 1e-6,
            "LREAL": 'Auto',
            "NSW": 0,
            "ENCUT": 550,
            "LAECHG": False,
            "NELM": 50,
            "NELMIN": 5,
            "LCHARG": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LVHAR": True,
            "KSPACING": None,
            "LWAVE": True,
            "KPAR": 2,
            "GGA": "PE",
            "ALGO": "Normal",
        }
