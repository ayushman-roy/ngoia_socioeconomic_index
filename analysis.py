import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

absolute_path = os.path.dirname(__file__)
mastersheet = pd.read_csv(os.path.join(absolute_path, "data/MASTERSHEET.csv"))

mastersheet.replace(["*", " ", "0"], np.nan, inplace=True)

# identifier columns
id_columns = ["dc", "st", "district", "state", "HDI", "mult_total"]
df_ids = mastersheet[id_columns].copy()

# new DataFrame for analysis by dropping the ID columns
df_analysis = mastersheet.drop(columns=id_columns)

df_analysis = df_analysis.apply(pd.to_numeric, errors='coerce')

spearman_corr = df_analysis.corr(method='spearman')
to_drop = set()
for col in spearman_corr.columns:
    for idx in spearman_corr.index:
        if col != idx and abs(spearman_corr.loc[idx, col]) > 0.9:
            to_drop.add(col)  

# drop highly correlated variables from the analysis DataFrame
df_analysis = df_analysis.drop(columns=to_drop)
print("Variables Removed:", to_drop)

# legend of the final variables used can be found @ LEGEND.xlsx
MASTERSHEET_postSpearman = os.path.join(absolute_path, "data/MASTERSHEET_after_Spearman.csv")
df_analysis.to_csv(MASTERSHEET_postSpearman, index=False)
