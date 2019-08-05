import argparse
import glob
import pandas as pd
import seaborn as sb
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory", required=True,
                    help="Path to input directory where asses results are")
parser.add_argument("-r", "--rounds-file", required=True,
                    help="File listing rounds ordered")
parser.add_argument("-o", "--output-report-file", default='report.pdf',
                    help="Path for output report file")

args = parser.parse_args()

# We need to read in vectors of assess results, merge them based on the
# vectors title and then produce the violin plots per round. This means
# that each asses result can contain iterations for the same round, so they need
# to be merged first.

acc_arrays = {}

# We only need to store rounds that are in the rounds file
rounds_df = pd.read_csv(args.rounds_file, header=None, names=['rounds'])

for asses_file in glob.glob("{}/sccaf_assess_*.txt".format(args.input_directory)):
    df = pd.read_csv(asses_file)
    if rounds_df['rounds'].str.contains(df.columns.array[0]).any():
        if df.columns.array[0] in acc_arrays:
            acc_arrays[df.columns.array[0]].append(df)
        else:
            acc_arrays[df.columns.array[0]] = df

df_merged = pd.DataFrame()
for round in rounds_df['rounds']:
    acc_array = acc_arrays[round]
    df_merged[round] = acc_array[acc_array.columns.array[0]]

# Now plot the merged data

fig, ax = pyplot.subplots(figsize=(9, 7))
sb.violinplot(ax=ax, y=df_merged)
pyplot.savefig('sccaf_assesment_accuracies.png')

