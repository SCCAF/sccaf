import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy.api as sc
import SCCAF as sf
import logging

import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file",
                    help="Path to input in AnnData or Loom")
parser.add_argument("-o", "--output-file", default='output.h5',
                    help="Path for output file")
parser.add_argument("-e", "--external-clustering-tsv",
                    help="Path to external clustering in TSV")
parser.add_argument("--optimise",
                    help="Not only run assessment, but also optimise the clustering",
                    action="store_true")
parser.add_argument("--skip-assessment", action="store_true",
                    help="If --optimise given, then this allows to optionally skip the original "
                         "assessment of the clustering")
parser.add_argument("-s", "--slot-for-existing-clustering",
                    help="Use clustering pre-computed in the input object, available in this slot of the object.")
parser.add_argument("-r", "--resolution", default=1.5, type=float,
                    help="Resolution for running clustering when no internal or external clustering is given.")
parser.add_argument("-a", "--min-accuracy", type=float,
                    help="Accuracy threshold for convergence of the optimisation procedure.")
parser.add_argument("-p", "--prefix",
                    help="Prefix for clustering labels", default="L1")

args = parser.parse_args()

if (not args.skip_assessment) and not (args.external_clustering_tsv or args.slot_for_existing_clustering):
    logging.error("Either --external-clustering-tsv or --slot-for-existing-clustering needs to be set "
                  "if the assessment is to be done.")
    exit(1)

ad = sc.read(args.input_file)

if args.external_clustering_tsv:
    cluster = pd.read_table(args.external_clustering_tsv, usecols=[0, 1], index_col=0, header=0)
    cluster.columns = ['CLUSTER']
    y = (pd.merge(ad.obs, cluster, how='left', left_index=True, right_index=True))['CLUSTER']
else:
    y = ad.obs[args.slot_for_existing_clustering]

raw = getattr(ad, 'raw')
if raw:
    X = raw.X
else:
    X = ad.X

if not args.skip_assessment:
    y_prob, y_pred, y_test, clf, cvsm, acc = sf.SCCAF_assessment(X, y)
    aucs = sf.plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
    plt.savefig('roc-curve.png')
    plt.close()

if args.optimise:
    if args.resolution:
        sc.tl.louvain(ad, resolution=args.resolution, key_added='{}_Round0'.format(args.prefix))
    else:
        # We use the predefined clustering (either internal or external).
        ad.obs['{}_Round0'.format(args.prefix)] = y

    sf.SCCAF_optimize_all(min_acc=args.min_accuracy, ad=ad, plot=False)
    #sc.pl.scatter(ad, base=args.visualisation, color=)
    y_prob, y_pred, y_test, clf, cvsm, acc = sf.SCCAF_assessment(X, ad.obs['{}_result'.format(args.prefix)])
    # Run 100 assement and look at the dispersion and average of acc, and with that try to pick the best round.
    aucs = sf.plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
    plt.savefig('optim.png')
    ad.write(args.output_file)
