# Input data

Place the following **11 input files** directly in this directory. The scripts use these exact filenames; no subdirectories are required.

| Filename | Role |
| --- | --- |
| `healthy_hematopoiesis_slim.h5ad` | Granja hematopoietic atlas, 35,038 cells and 50,000 peaks |
| `buenrostro2018.h5ad` | Buenrostro 2018 processed scATAC-seq |
| `andy_hematopoiesis.h5ad` | Zeng et al. HSC xenograft multiome input |
| `peak_to_catlas_programme.tsv.gz` | Granja peak-to-CATLAS programme assignment |
| `F3E_buenrostro_catlas_peak_assignment.tsv.gz` | Buenrostro peak-to-CATLAS programme assignment |
| `motifs.tsv` | HSC JASPAR motif names |
| `motif_matches.mtx` | HSC peak-by-motif match matrix |
| `peaks.txt` | Peaks corresponding to the motif matrix |
| `peak_gc_and_hsc_accessibility.tsv.gz` | GC and accessibility covariates for motif tests |
| `family_definitions.tsv` | Motif-family assignments |
| `HSCII_vs_HSCI_ATAC_DA_Markers.csv` | Published HSC-II vs HSC-I ATAC markers |

Approximate total local size: **696 MB**. These files are excluded from ordinary Git commits and are not included with the code release.

**Zenodo:** Pending deposit and redistribution review. Add the actual DOI, source accessions, download procedure, checksums, and licenses here only once verified. The prepared data derive from published projects, and their redistribution must comply with the original terms and attribution requirements.
