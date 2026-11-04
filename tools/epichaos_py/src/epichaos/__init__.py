"""epiCHAOS: Python reimplementation of the epiCHAOS R package.

Primary (AnnData) API — modifies adata in place, scanpy-style:
    import epichaos
    epichaos.epiCHAOS(adata, group_by="cell_type")
    epichaos.transCHAOS(adata, group_by="cluster")

Low-level (NumPy) API:
    from epichaos.core import compute_eITH, compute_tITH
    from epichaos.utils import create_group_matrices
"""

from epichaos.anndata_api import epiCHAOS, transCHAOS
from epichaos.core import compute_eITH, compute_tITH
from epichaos.utils import create_group_matrices
from epichaos.stats import compare_groups, compute_diff_eICH
from epichaos.regions import compare_regions
from epichaos import plotting

__version__ = "0.1.0"
__all__ = [
    "epiCHAOS",
    "transCHAOS",
    "compute_eITH",
    "compute_tITH",
    "create_group_matrices",
    "compare_groups",
    "compute_diff_eICH",
    "compare_regions",
    "plotting",
]
