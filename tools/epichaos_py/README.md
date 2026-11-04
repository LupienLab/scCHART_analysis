# epichaos-py

Python reimplementation of the [epiCHAOS R package](https://github.com/CompEpigen/epiCHAOS) (Kelly et al., *Genome Biology* 2024). Quantifies cell-to-cell epigenetic heterogeneity in single-cell epigenomics data (scATAC-seq, scChIP-seq, etc.) using a pairwise Jaccard-based metric.

## Install

Requires Python ≥ 3.11. Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install epichaos
```

Or with pip:

```bash
pip install epichaos
```

For development:

```bash
git clone https://github.com/your-org/epichaos-py
cd epichaos-py
uv sync --extra dev
```

## Quick start

```python
import scanpy as sc
import epichaos

# Load your single-cell data (cells × peaks binary matrix)
adata = sc.read_h5ad("my_scatac.h5ad")

# Compute epiCHAOS scores — results written into adata.obs and adata.uns
epichaos.epiCHAOS(adata, group_by="cell_type")

# Per-cell scores
print(adata.obs[["cell_type", "epiCHAOS_het_adj"]].drop_duplicates())

# Group-level summary
print(adata.uns["epiCHAOS"])

# Plot
fig = epichaos.plotting.plot_scores(adata, key_added="epiCHAOS")
fig.savefig("epiCHAOS_scores.png", bbox_inches="tight")
```

For scRNA-seq (transcriptional heterogeneity):

```python
epichaos.transCHAOS(adata_rna, group_by="cluster")
print(adata_rna.uns["transCHAOS"])
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_by` | — | `adata.obs` column defining cell groups (required) |
| `n` | 50 | Max cells to subsample per group |
| `m` | 20 | Min cells required to score a group |
| `subsample` | 1 | Pseudoreplicates (1 = compute once) |
| `key_added` | `"epiCHAOS"` | Key for `adata.obs` columns and `adata.uns` |

Scores are written as:
- `adata.obs[f"{key_added}_het_adj"]` — sparsity-adjusted score per cell (primary metric)
- `adata.obs[f"{key_added}_het_raw"]` — raw score before adjustment
- `adata.uns[key_added]` — group-level DataFrame

## Testing

```bash
# Unit tests (no data required)
pytest tests/test_jaccard.py -v

# Integration tests (requires data setup — see below)
pytest tests/test_integration.py -v
```

Integration tests compare Python output against R reference scores and require the bundled R datasets to be exported first:

```bash
# 1. Export R data and compute reference scores (needs R + epiCHAOS R package)
Rscript scripts/export_r_data.R

# 2. Convert to AnnData h5ad format
python scripts/convert_to_anndata.py

# 3. Run all tests
pytest -v
```

Integration tests are automatically skipped if the data files are absent.

## Algorithm

epiCHAOS scores are computed per group:
1. Subsample to ≤ `n` cells; remove zero-coverage cells.
2. Compute mean pairwise **centered Jaccard** (`J − E[J]`) across all cell pairs.
3. Regress raw scores on mean coverage → residuals → `het_adj`.
4. Both `het_raw` and `het_adj` are min-max normalized and negated so **higher = more heterogeneous**.

The implementation matches the R package exactly (validated to machine precision on the bundled hematopoiesis dataset). See `IMPLEMENTATION_LOG.md` for details.

## Citation

If you use this package, please cite the original R package:

> Kelly, K., Scherer, M., Braun, M.M. et al. *epiCHAOS: a metric to quantify epigenomic heterogeneity in single-cell data.* Genome Biol 25, 305 (2024). https://doi.org/10.1186/s13059-024-03446-w
