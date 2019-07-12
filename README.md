# SCCAF: Single Cell Clustering Assessment Framework

Single Cell Clustering Assessment Framework (SCCAF) is a novel method for automated identification of putative cell types from single cell RNA-seq (scRNA-seq) data. By iteratively applying clustering and a machine learning approach to gene expression profiles of a given set of cells, SCCAF simultaneously identifies distinct cell groups and a weighted list of feature genes for each group. The feature genes, which are overexpressed in the particular cell group, jointly discriminate the given cell group from other cells. Each such group of cells corresponds to a putative cell type or state, characterised by the feature genes as markers.

# Requirements

This package requires Python 3 and pip3 for installation, which will take care of dependencies.

# Installation

You can install SCCAF with pip:

```
pip install sccaf
```

Alternatively, for the latest version, clone this repo and go into its directory, then execute `pip3 install .`:

```
git clone https://github.com/SCCAF/sccaf
cd sccaf
# you might want to create a virtualenv for SCCAF before installing
pip3 install .
```

if your python environment is configured for python 3, then you should be able to replace python3 for just python (although pip3 needs to be kept). In time this will be simplified by a simple pip call.

# Command line runs

## Use with pre-clustered `anndata` object

The main method of SCCAF can be applied directly to an [anndata](https://anndata.readthedocs.io/en/stable/) (AnnData is the main data format used by [Scanpy](https://scanpy.readthedocs.io/en/stable/)) object in Python provided that it has been previously clustered and that the data is not too batchy (or that it has been batch corrected):

```python
from SCCAF import SCCAF_optimize_all
import scanpy.sc as sc

ad = sc.read("path-to-clusterised-and-umapped-anndata-file")
# Set the initial starting point, assuming that cells were clusterised through louvain.
ad.obs['L1_Round0'] = tm.obs['louvain']
SCCAF_optimize_all(ad=tm, plot=False, min_acc=0.96)
```

in the above run, all changes will be left on the `ad` anndata object and no plots
will be generated. If you want to see the plots (blocking the progress until you close them)
then remove the `plots=False`.


Within the anndata object, assignments of cells to clusters will be left in `ad.obs['L1_Round<roundNumber>']`.
