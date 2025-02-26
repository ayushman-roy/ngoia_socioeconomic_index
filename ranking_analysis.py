import os
import pandas as pd
import numpy as np
from scipy import stats

absolute_path = os.path.dirname(__file__)
master_df = pd.read_csv(os.path.join(absolute_path, "data/mastersheet.csv"))
results_df = pd.read_csv(os.path.join(absolute_path, "data/results.csv"))

scores_df = results_df.iloc[:, 5:5+36] 
ranks_df = results_df.iloc[:, -36:]
hdi_df = results_df.iloc[:, [4]]
hdi_rank_df = hdi_df.iloc[:, 0].rank(ascending=False).to_frame(name='rank')

# compute Pearson correlation
corr_scores = scores_df.apply(lambda x: x.corr(hdi_df.iloc[:, 0], method='pearson'))
corr_ranks = ranks_df.apply(lambda x: x.corr(hdi_rank_df.iloc[:, 0], method='pearson'))

# best and worst correlations with HDI
best_scores = corr_scores.nlargest(5, keep='all')
worst_scores = corr_scores.nsmallest(5, keep='all')
best_ranks = corr_ranks.nlargest(5, keep='all')
worst_ranks = corr_ranks.nsmallest(5, keep='all')

correlation_results = {
    "best_scores": best_scores,
    "worst_scores": worst_scores,
    "best_ranks": best_ranks,
    "worst_ranks": worst_ranks
}

print("Best HDI-Score Correlations:\n", best_scores)
print("Worst HDI-Score Correlations:\n", worst_scores)
print("\nBest HDI-Rank Correlations:\n",best_ranks )
print("Worst HDI-Rank Correlations:\n", worst_ranks)

# load saved pairs from file
with open("data/spearman_cols.txt", "r") as file:
    spearman_corr_pairs = [line.strip().split(",") for line in file.readlines()]

print("Spearman Correlation Pairs:", spearman_corr_pairs)

# quartile analysis

# [SELECT] RANKING AND VARIABLES FOR STATISTICS HERE
ranking = "3_socioeconomic_ranks"   
var_list = ["sex_ratio", "men_tobbaco", "marst_1"] 

# new dataframe with selected variables
selected_vars_df = pd.DataFrame()
for var in var_list:
   if var in master_df.columns:
       selected_vars_df[var] = master_df[var]

quartile_analysis = results_df.join(selected_vars_df)

# compute quartiles on ranking 
quartile_analysis['quartile'] = pd.qcut(quartile_analysis[ranking], q=4, labels=[1, 2, 3, 4])
results = []

# quartile statistics  
for q in range(1, 5):
    quartile_data = quartile_analysis[quartile_analysis['quartile'] == q]
    results.append(f"Quartile {q} Statistics:\n")
    
    for var in var_list:
        if var in quartile_data.columns:
            mean_val = quartile_data[var].mean()
            median_val = quartile_data[var].median()
            mode_val = stats.mode(quartile_data[var], nan_policy='omit').mode
            std_val = quartile_data[var].std()
            
            results.append(f"Variable: {var}\n")
            results.append(f"Mean: {mean_val}\n")
            results.append(f"Median: {median_val}\n")
            results.append(f"Mode: {mode_val}\n")
            results.append(f"Standard Deviation: {std_val}\n\n")

# export results to a .txt file
with open(f"output/rank_statistics/{ranking}_statistics.txt", "w") as f:
    f.writelines(results)

print(f"Analysis Complete @ output/rank_statistics/{ranking}_statistics.txt.")
