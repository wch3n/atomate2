"""Flows for calculating redox potentials for adsorbates on a specific substrate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker, OutputReference

from atomate2.utils.file_client import FileClient
from atomate2.vasp.run import JobType
from atomate2.common.files import get_zfile
from atomate2.common.jobs.redox import (
    adsorb_molecule,
    dynmat_calculation,
    molecule_calculation,
    substrate_supercell_calculation,
    adsorbate_supercell_calculation,
)

from atomate2.vasp.jobs.redox import (
    get_structure_from_prv_calc,
)
from atomate2.vasp.flows.core import (
    DoubleRelaxMaker,
)
from atomate2.vasp.jobs.core import (
    HSEStaticMaker,
    HSERelaxMaker,
    StaticMaker,
    RelaxMaker,
)
from atomate2.vasp.sets.core import (
    HSEStaticSetGenerator,
    StaticSetGenerator,
)
from atomate2.vasp.sets.redox import (
    SlabRelaxSetGenerator,
    SlabStaticSetGenerator,
)
from pymatgen.core import Structure

from custodian.vasp.handlers import (
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    PositiveEnergyErrorHandler,
    StdErrHandler,
    VaspErrorHandler,
    UnconvergedErrorHandler,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

REDOX_PRESET = {}
REDOX_PRESET['OER'] = {0:{"reactant": "H2O", "ads": "OH",  "gas": "H", "e": 1}, 
        1:{"reactant": None,  "ads": "O",   "gas": "H", "e": 1},
        2:{"reactant": "H2O", "ads": "OOH", "gas": "H", "e": 1},
        3:{"reactant": None,  "ads": None,  "gas": ("O2", "H"), "e": 1}}

REDOX_PRESET['OER_bi'] = {0:{"reactant": "H2O", "ads": "OH",  "gas": "H", "e": 1, "ads_site":'a'}, 
        1:{"reactant": None,  "ads": "O",   "gas": "H", "e": 1, "ads_site": 'a'},
        2:{"reactant": "H2O", "ads": "H", "gas": "O2", "e": 1, "ads_site":'b'},
        3:{"reactant": None,  "ads": None,  "gas": "H", "e": 1}}

REDOX_PRESET['CO2RR_1'] = {0:{"reactant": ("CO2","H"), "ads": "COOH", "gas":None, "e": 1}, 
        }
REDOX_PRESET['CO2RR_2'] = {0:{"reactant": ("CO2","H"), "ads": "HCOO", "gas":None, "e": 1}, 
        }
REDOX_PRESET['HER'] = {0:{"reactant": "H", "ads": "H", "gas":None, "e": 1}, 
        }

INCAR_SETTINGS = {
    "ALGO": "Normal",
    "NELM": 30,
    "NSW": 50,
}

INCAR_SETTINGS_MOL = INCAR_SETTINGS.copy()
INCAR_SETTINGS_MOL['KPAR'] = 1
INCAR_SETTINGS_DYNMAT = INCAR_SETTINGS_MOL.copy()
INCAR_SETTINGS_DYNMAT['IBRION'] = 7
INCAR_SETTINGS_DYNMAT['NSW'] = None
INCAR_SETTINGS_DYNMAT['LREAL'] = False

KPOINT_SETTINGS = {"reciprocal_density": 200}
KPOINT_SETTINGS_G = {"reciprocal_density": 1}

SLAB_RELAX_GENERATOR = SlabRelaxSetGenerator(
    user_incar_settings=INCAR_SETTINGS,
    user_kpoints_settings=KPOINT_SETTINGS,
)
MOLECULE_RELAX_GENERATOR = SlabRelaxSetGenerator(
    user_incar_settings=INCAR_SETTINGS_MOL,
    user_kpoints_settings=KPOINT_SETTINGS_G,
)
DYNMAT_GENERATOR = SlabStaticSetGenerator(
    user_incar_settings=INCAR_SETTINGS_DYNMAT,
    user_kpoints_settings=KPOINT_SETTINGS_G,
)

RUN_VASP_KWARGS = { "handlers": (
    VaspErrorHandler(),
    #MeshSymmetryErrorHandler(),
    #PositiveEnergyErrorHandler(),
    #FrozenJobErrorHandler(),
    #StdErrHandler(),
    #LargeSigmaHandler(),
    #IncorrectSmearingHandler(),
    UnconvergedErrorHandler(),
    ),
    "custodian_kwargs": {"terminate_on_nonzero_returncode": False}
}

#RUN_VASP_KWARGS = {"job_type":"direct"}

def flatten_list(nested_list: list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list,tuple)):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

@dataclass
class RedoxPotentialMaker(Maker):
    molecule_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=MOLECULE_RELAX_GENERATOR,
            run_vasp_kwargs=RUN_VASP_KWARGS,
        ),
    )
    substrate_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SLAB_RELAX_GENERATOR,
            run_vasp_kwargs=RUN_VASP_KWARGS,
        ),
    )
    adsorbate_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SLAB_RELAX_GENERATOR,
            run_vasp_kwargs=RUN_VASP_KWARGS,
        ),
    )
    dynmat_maker: BaseVaspMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=DYNMAT_GENERATOR,
            run_vasp_kwargs=RUN_VASP_KWARGS,
        ),
    )

    sc_mat: list | npt.NDArray | None = None
    name: str = "redox potential"

    def make(
        self,
        redox_type: str,
        substrate_poscar: str | Path,
        supercell_matrix: npt.NDArray | None = None,
        thermodynamic_corrections: dict | None = None,
        substrate_supercell_dir: str | Path | None = None,
        anchor_site: int | list[int] | None = 0,
        anchor_sites: dict | None = None,
        height: float = 1.8,
        theta: float | None = None,
        offset: list[float] | None = None,
        offset_subs: dict | None = None,
        fix_below: float | None = None,
        skip: list[str] | str | None = None,
        functional: str | None = None,
        vdw: str | None = None,
        dynmat: bool = False,
    ) -> Flow:

        jobs = []
        redox = REDOX_PRESET[redox_type]

        reactants = flatten_list([k for k in [j for j in [redox[i]['reactant'] 
            for i in redox] if j is not None]])
        gas_prods = flatten_list([k for k in [j for j in [redox[i]['gas'] 
            for i in redox] if j is not None]])
        
        skip_mol = True if isinstance(skip,(str, list)) and 'molecule' in skip else False
    
        if not skip_mol:
            for mol_name in set(reactants + gas_prods):
                mol_job = molecule_calculation(
                    molecule_name = mol_name,
                    relax_maker=self.molecule_relax_maker,
                )
                jobs.append(mol_job)
                if dynmat:
                    dynmat_job = dynmat_calculation(
                        structure=mol_job.output['mol_struct'],
                        static_maker=self.dynmat_maker,
                        prv_vasp_dir=mol_job.output['dir_name'],
                    )
                    jobs.append(dynmat_job) 

        substrate = Structure.from_file(substrate_poscar)
        
        if substrate_supercell_dir is None:
            sc_job = substrate_supercell_calculation(
                uc_structure=substrate,
                relax_maker=self.substrate_relax_maker,
                sc_mat=supercell_matrix,
                fix_below = fix_below,
            )
        else:
            sc_job = get_structure_from_prv_calc(
                prv_calc_dir=substrate_supercell_dir
            )
        jobs.append(sc_job)
       
        for _stage in redox.keys():
            ads = redox[_stage]['ads']
            if ads is not None:
                if 'ads_site' in redox[_stage]:
                    anchor_site = anchor_sites[redox[_stage]['ads_site']]
                adsorb_job = adsorb_molecule(
                    molecule_name=ads,
                    substrate=sc_job.output['sc_struct'],
                    anchor_site=anchor_site,
                    height=height,
                    theta=theta, 
                    offset=offset,
                    offset_subs=offset_subs,
                    fix_below = fix_below,
                )
                jobs.append(adsorb_job)

                ads_sc_job = adsorbate_supercell_calculation(
                    adsorbate_structure=adsorb_job.output,
                    relax_maker=self.adsorbate_relax_maker,
                )
                jobs.append(ads_sc_job)

        return Flow(jobs, name=self.name)
