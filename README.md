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

## Use with pre-clustered `anndata` object in the [SCANPY](https://scanpy.readthedocs.io/en/stable/) package

The main method of SCCAF can be applied directly to an [anndata](https://anndata.readthedocs.io/en/stable/) (AnnData is the main data format used by [Scanpy](https://scanpy.readthedocs.io/en/stable/)) object in Python. 

**Before applying SCCAF, please make sure the doublets have been excluded and the batch effect has been effectively regressed.**

## Assessment of the quality of a clustering

Given a clustering stored in an anndata object `adata` under the key `louvain`, we would like to understand the quality (discrimination between clusters) with SCCAF:

```python
from SCCAF import SCCAF_assessment, plot_roc
import scanpy as sc

adata = sc.read("path-to-clusterised-and-umapped-anndata-file")
y_prob, y_pred, y_test, clf, cvsm, acc = SCCAF_assessment(adata.X, adata.obs['louvain'], n=100)
```

returned accuracy is in the `acc` variable.

The ROC curve can be plotted:

```python
import matplotlib.pyplot as plt

plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
plt.show()
```

Higher accuracy indicate better discrimination. And the ROC curve shows the problematic clusters. 

## Optimize an over-clustering

Given an over-clustered result, SCCAF optimize the clustering by merging the cell clusters that cannot be discriminated by machine learning:

```python

# The batch effect MUST be regressed before applying SCCAF
adata = sc.read("path-to-clusterised-and-umapped-anndata-file")

# An initial over-clustering needs to be assigned in consistent with the prefix for the optimization.
# i.e., the optimization prefix is `L2`, the starting point of the optimization of `%s_Round0`%prefix, which is `L2_Round0`.

sc.tl.louvain(adata, resolution=1.5, key_added='L2_Round0')
# i.e., we aim to achieve an accuracy >90% for the whole dataset, optimize based on the PCA space:
SCCAF_optimize_all(ad=adata, plot=False, min_acc=0.9, prefix = 'L2', use='pca')
```

in the above run, all changes will be left on the `adata` anndata object and no plots
will be generated. If you want to see the plots (blocking the progress until you close them)
then remove the `plots=False`.


Within the anndata object, assignments of cells to clusters will be left in `adata.obs['<prefix>_Round<roundNumber>']`.
