import csv, io, ast
import pandas as pd
import numpy as np
from itertools import groupby
import scarp_analysis as sa


# .CSV file content extraction
data_tabs = []
with io.open('./data/data.csv', newline='') as csvfile:
    csv_ROWs = csv.reader(csvfile, delimiter=',', quotechar='|')

    # Skip header
    header_line = next(csv_ROWs)        # FID,Profile,Distance,Scarps,Depth,X_Coord,Y_Coord

    included_cols = [0, 1, 2, 3, 4, 5, 6]     # select columns to be studied
    header_line = [header_line[i] for i in included_cols]

    for row in csv_ROWs:
        # Convert row values from string to their own type
        conv_TMP = [ast.literal_eval(row[i]) for i in included_cols]
        data_tabs.append(conv_TMP)

# Convert to array
data_tabs = np.asarray(data_tabs)

# Extract columns of data
class_dict = {}
for i in range(len(header_line)):
    class_dict[header_line[i]] = data_tabs[:, i]

# Reconstruct data matrix
dataFrame = pd.DataFrame(data=class_dict, columns=header_line)

# Get IDs of profiles represented in data
p_ids = []
for key, group in groupby(list(class_dict['Profile'])):
    p_ids.append(int(list(group)[0]))

# Organize frames into different profiles
data_dict = {}
for i in p_ids:
    data_dict[i] = dataFrame.loc[dataFrame['Profile'] == i]

# Initialize quantifier
quant = sa.quantifier()

# Run statistical analysis
quant.run_analysis(data=data_dict, profiles=p_ids)

# Write results
quant.write_outputs(out_path="./results/output.csv")

# Plot
quant.plot_heaves(limit=1000)
quant.plot_strain(transects=list(np.arange(0,500)))
quant.plot_histogram(var=quant.heaves_W, bin_size=40)
quant.plot_ratios(var1=quant.heaves_W, var2=quant.heaves_rsh_W, bin_size=50)
quant.plot_dualscatter(varx=[quant.W_strain, quant.E_strain],
                       vary=[quant.gaps_W_areas_sum, quant.gaps_E_areas_sum])