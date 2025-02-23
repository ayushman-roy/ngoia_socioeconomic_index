import pandas as pd
import os

absolute_path = os.path.dirname(__file__)

mastersheet = pd.read_csv(os.path.join(absolute_path, "data/MASTERSHEET_after_Spearman.csv"))

# * make ideal data point using legend