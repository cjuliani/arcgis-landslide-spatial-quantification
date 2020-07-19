import arcpy
import csv, io, ast
import numpy as np
import pandas as pd
from itertools import groupby

# .CSV file content extraction
data_tabs = []
with io.open('./results/output.csv', newline='') as csvfile:
    csv_ROWs = csv.reader(csvfile, delimiter=',', quotechar='|')

    # Skip header
    header_line = next(csv_ROWs)        # FID,Profile,Distance,Scarps,Depth,X_Coord,Y_Coord

    included_cols = [0, 1, 2, 3]     # select columns to be studied
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


# Database update
gridName = 'transect_points'		# represent the vertex of transects drawn in ArcGIS
tableInput = gridName +'.dbf'
fldName = 'Gaps'
to_update = np.array(dataFrame[fldName])

# Updating
print("Updating the tables...")
fields = ['FID',fldName]
i = 0
with arcpy.da.UpdateCursor(gridName, fields) as cursor:
	for row in cursor:
		if row[0] <= len(data_dict[fldName]):
			try:
				row[1] = to_update[i]
				cursor.updateRow(row)
			except:
				row[1] = 0
				cursor.updateRow(row)
			i = i+1
		else:
			break