"""Jobs for redox calculations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from math import cos, sin, pi
from jobflow import Flow, Response, job
from pymatgen.core import Structure, Molecule, Lattice
from pymatgen.core.sites import Site

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from emmet.core.tasks import TaskDoc
    from numpy.typing import NDArray

    from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker

logger = logging.getLogger(__name__)

MOLECULES = {'OH': Molecule(['O', 'H'], [(0,0,0),(0,0,1)]),
             'O': Molecule(['O'], [(0,0,0)]),
             'OOH': Molecule(['O','O','H'], [(-0.02,-0.02,0),(0.5,0.2,1.0),(-0.4,0.8,1.5)]),
             'H2': Molecule(['H', 'H'], [(0,0,0),(0,0,0.74)]),
             'O2': Molecule(['O', 'O'], [(0,0,0),(0,0,1.16)]),
             'H2O': Molecule(['O','H','H'], [(0,0,0),(0.76,0.59,0),(-0.76,0.59,0)]),
             'CO2': Molecule(['C','O','O'], [(0,0,0),(-1.2,0,0),(1.2,0,0)]),
             'COOH': Molecule(['C','O','O','H'], [(0,0,0),(1.1,0.0,0.5),(-1.2,0.0,0.5),(-1.2,0.0,1.5)]),
             'HCOO': Molecule(['O','C','O','H'], [(0,0,0),(1.2,0.1,0.8),(2.4,0.1,0.0),(1.2,0.1,1.8)]),
             'H': Molecule(['H'],[(0,0,0)]),
}

CUBIC_LAT = [[15, 0, 0], [0, 15, 0], [0, 0, 15]]

@job
def adsorb_molecule(
    molecule_name: str,
    substrate: Structure | None,
    anchor_site: int | list[int] | None,
    height: float = 1.8,
    theta: float | None = None,
    offset: list[float] | None = None,
    offset_subs: dict | None = None,
    fix_below: float | None = None,
) -> Structure:
        
    mol = MOLECULES[molecule_name].copy()
    if offset_subs:
        for idx, shift_vector in offset_subs.items():
            substrate[int(idx)].frac_coords += shift_vector
    if isinstance(anchor_site, int):
        anchor_site = [anchor_site]
    anchor_coords = np.mean([substrate.sites[i].coords for i in anchor_site], axis=0)
    if theta:
        theta_rad  = theta/180.0*pi
        rot = np.matrix([[cos(theta_rad), -sin(theta_rad), 0],
                     [sin(theta_rad),  cos(theta_rad), 0],
                     [0,               0,              1]])
        cart_rotated = np.matrix(mol.cart_coords) * rot
        for idx in range(len(mol)):
            site = mol[idx]
            mol[idx] = Site(site.species, cart_rotated[idx].A1, properties=site.properties, label=site.label)
    if offset:
        cart_shifted = np.array(mol.cart_coords) + np.array(offset)
        for idx in range(len(mol)):
            site = mol[idx]
            mol[idx] = Site(site.species, cart_shifted[idx], properties=site.properties, label=site.label)
    for atom in mol:
        substrate.append(atom.species, 
                         coords=anchor_coords + [0,0,height] + atom.coords,
                         coords_are_cartesian = True, 
                         validate_proximity = True)
    if fix_below:
        selective_dynamics = []
        for i in substrate:
            if i.z < fix_below:
                selective_dynamics.append([False, False, False])
            else:
                selective_dynamics.append([True, True, True])
        substrate.add_site_property("selective_dynamics", selective_dynamics)
        
    return substrate

@job
def dynmat_calculation(
    structure: Structure,
    static_maker: StaticMaker,
    prv_vasp_dir: str | Path | None = None,
) -> Response:

    logger.info("Running dynamical matrix calculation. Running...")
    static_job = static_maker.make(structure, prv_vasp_dir)
    static_job.name = "dynamical matrix"
    static_output: TaskDoc = static_job.output
    summary = {
        "structure": structure,
        "entry": static_output.entry,
        "dir_name": static_output.dir_name,
        "uuid": static_output.uuid,
    }
    flow = Flow([static_job], output=summary)
    return Response(replace=flow)

@job
def molecule_calculation(
    molecule_name: str,
    relax_maker: RelaxMaker,
) -> Response:

    logger.info("Running molecule calculation. Running...")
    if molecule_name == 'H':
        molecule_name = 'H2'
    mol = MOLECULES[molecule_name]
    molecule_structure = Structure(Lattice(CUBIC_LAT), mol.species, 
        mol.cart_coords, coords_are_cartesian=True)   
    relax_job = relax_maker.make(molecule_structure)
    relax_job.name = 'molecule relax'
    relax_output: TaskDoc = relax_job.output
    summary = {
        "mol_entry": relax_output.entry,
        "mol_struct": relax_output.structure,
        "dir_name": relax_output.dir_name,
        "uuid": relax_job.uuid,
    }
    flow = Flow([relax_job], output=summary)
    return Response(replace=flow)


@job
def substrate_supercell_calculation(
    uc_structure: Structure,
    relax_maker: RelaxMaker,
    sc_mat: NDArray | list | None = None,
    fix_below: float | None = None,
) -> Response:

    logger.info("Running substrate supercell calculation. Running...")
    sc_structure = uc_structure if sc_mat is None else uc_structure.make_supercell(sc_mat)
    if fix_below:
        selective_dynamics = []
        for i in sc_structure:
            if i.z < fix_below:
                selective_dynamics.append([False, False, False])
            else:
                selective_dynamics.append([True, True, True])
        sc_structure.add_site_property("selective_dynamics", selective_dynamics)

    relax_job = relax_maker.make(sc_structure)
    relax_job.name = 'substrate relax'
    relax_output: TaskDoc = relax_job.output
    summary = {
        "uc_structure": uc_structure,
        "sc_entry": relax_output.entry,
        "sc_struct": relax_output.structure,
        "sc_mat": sc_mat,
        "dir_name": relax_output.dir_name,
        "uuid": relax_job.uuid,
    }
    flow = Flow([relax_job], output=summary)
    return Response(replace=flow)

@job
def adsorbate_supercell_calculation(
    adsorbate_structure: Structure,
    relax_maker: RelaxMaker,
) -> Response:

    logger.info("Running adsorbate supercell calculation. Running...")
    relax_job = relax_maker.make(adsorbate_structure)
    relax_job.name = 'adsorbate relax'
    relax_output: TaskDoc = relax_job.output
    summary = {
        "sc_entry": relax_output.entry,
        "sc_struct": relax_output.structure,
        "dir_name": relax_output.dir_name,
        "uuid": relax_job.uuid,
    }
    flow = Flow([relax_job], output=summary)
    return Response(replace=flow)
