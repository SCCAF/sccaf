import pandas as pd
import matplotlib
import scanpy.api as sc
import SCCAF as sf
import logging
import argparse
from numpy import arange

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file",
                    help="Path to input in AnnData or Loom", required=True)
parser.add_argument("-o", "--output-table",
                    help="Path for output file with table of accuracy and other metrics (required if iterations > 1)")
parser.add_argument("-e", "--external-clustering-tsv",
                    help="Path to external clustering in TSV")
parser.add_argument("-s", "--slot-for-existing-clustering",
                    help="Use clustering pre-computed in the input object, available in this slot of the object.")
parser.add_argument("--iterations",
                    help="Number of times to iterate the assesment to build distributions of accuracies", type=int,
                    default=1)
parser.add_argument("-c", "--cores",
                    help="Number of processors to use", type=int, default=1)
parser.add_argument("--use-pca",
                    help="Use PCA components for assesment (needs to be available in the ann data ) (default: False)",
                    action='store_true')

args = parser.parse_args()

if not (args.external_clustering_tsv and args.slot_for_existing_clustering):
    logging.error("Either --external-clustering-tsv or --slot-for-existing-clustering needs to be set ")
    exit(1)

ad = sc.read(args.input_file)

if args.external_clustering_tsv:
    cluster = pd.read_table(args.external_clustering_tsv, usecols=[0, 1], index_col=0, header=0)
    cluster.columns = ['CLUSTER']
    y = (pd.merge(ad.obs, cluster, how='left', left_index=True, right_index=True))['CLUSTER']
    column_name = "external_clustering"
else:
    y = ad.obs[args.slot_for_existing_clustering]
    column_name = args.slot_for_existing_clustering
accs_out_fn = "{}.csv".format(column_name)

if args.output_table:
    accs_out_fn = args.output_table

raw = getattr(ad, 'raw')
if raw:
    X = raw.X
else:
    X = ad.X

if args.use_pca:
    if 'X_pca' in ad.obsm_keys():
        X = ad.obsm['X_pca']
    else:
        logging.warning("PCA mode activated, but ann data object doesn't have PCA loadings, using previous default")


accs =[]
for i in arange(args.iterations):
    y_prob, y_pred, y_test, clf, cvsm, acc = sf.SCCAF_assessment(X, y, n_jobs=args.cores)
    if args.iterations == 1:
        aucs = sf.plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
        plt.savefig('roc-curve.png')
        plt.close()
    else:
        accs.append(acc)

if args.iterations == 1:
    pd.DataFrame(accs, columns=[column_name]).to_csv(accs_out_fn, index=False)
