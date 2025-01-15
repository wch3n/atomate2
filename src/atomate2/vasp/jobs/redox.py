"""Jobs for defect calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from emmet.core.tasks import TaskDoc
from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)

@job
def get_structure_from_prv_calc(
    prv_calc_dir: str | Path
) -> dict:

    prv_calc_dir = prv_calc_dir.split(":")[-1]
    logger.info(prv_calc_dir)
    task_doc = TaskDoc.from_directory(prv_calc_dir)
    return {
        "sc_struct": task_doc.structure,
        "dir_name": prv_calc_dir
    }
