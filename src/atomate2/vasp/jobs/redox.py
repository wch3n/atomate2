"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import job
from pymatgen.io.vasp import Vasprun
from atomate2.utils.file_client import FileClient
from atomate2.common.files import get_zfile

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)

@job
def get_structure_from_prv_calc(
    prv_calc_dir: str | Path
) -> dict:

    fc = FileClient()
    prv_calc_dir = prv_calc_dir.split(":")[-1]
    logger.info(prv_calc_dir)
    files = fc.listdir(prv_calc_dir)
    vasprun_file = Path(prv_calc_dir) / get_zfile(files, "vasprun.xml")
    logger.info(prv_calc_dir)
    vasprun = Vasprun(vasprun_file)
    return {
        "sc_struct": vasprun.final_structure,
        "dir_name": prv_calc_dir
    }
