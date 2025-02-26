import pandas as pd
import os

absolute_path = os.path.dirname(__file__)

legend = pd.read_csv(os.path.join(absolute_path, "data/LEGEND.csv"))
mastersheet = pd.read_csv(os.path.join(absolute_path, "data/MASTERSHEET.csv"))

new_row = pd.Series(index=mastersheet.columns, dtype='object')

for _, row in legend.iterrows():
    var_name = row["Variable"]
    ideal = row["Ideal"]
    
    if var_name in mastersheet.columns:  # ensure variable exists in mastersheet
        if pd.notna(ideal):
            if str(ideal).lower() == "min":
                new_row[var_name] = mastersheet[var_name].min()
            elif str(ideal).lower() == "max":
                new_row[var_name] = mastersheet[var_name].max()
            elif pd.to_numeric(ideal, errors='coerce') is not None:  # Numeric check
                new_row[var_name] = ideal
            else:
                new_row[var_name] = ""
        else:
            new_row[var_name] = ""

new_row["district"] = "IDEAL"
new_row["state"] = "IDEAL"

new_row.to_csv("data/ideal_row.txt", sep='\t', index=True, header=False)
mastersheet = pd.concat([mastersheet, pd.DataFrame([new_row])], ignore_index=True)
mastersheet.to_csv("data/mastersheet.csv", index=False)

print("Ideal District added @ MASTERSHEET")
