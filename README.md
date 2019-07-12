This package implements ....

# Requirements

This package requires Python 3, setuptools and pip3 for installation.
We will provide eventually a pip installable version.

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

## Use with pre-clusterised `anndata` object

The main method of SCCAF can be applied directly to an anndata object in Python:

```
from SCCAF import SCCAF_optimize_all
import scanpy.sc as sc

ad = sc.read("path-to-clusterised-and-umapped-anndata-file")
# Set the initial starting point, assuming that cells where clusterised through louvain.
ad.obs['L1_Round0'] = tm.obs['louvain']
SCCAF_optimize_all(ad=tm, plot=False, min_acc=0.96)
```

in the above run, all changes will be left on the `ad` anndata object and no plots
will be generated. If you want to see the plots (blocking the progress until you close them)
then remove the `plots=False`.

Within the anndata object, assignments of cells to clusters will be left in `ad.obs['L1_Round<roundNumber>']`.
