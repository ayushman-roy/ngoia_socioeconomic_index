import pandas as pd
import os
import numpy as np

absolute_path = os.path.dirname(__file__)
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

print("Best score correlations (Scores):")
print(best_scores)
print("Worst score correlations (Scores):")
print(worst_scores)
print("Best rank correlations (Ranks):")
print(best_ranks)
print("Worst rank correlations (Ranks):")
print(worst_ranks)


# load saved pairs from file
with open("data/spearman_cols.txt", "r") as file:
    spearman_corr_pairs = [line.strip().split(",") for line in file.readlines()]

print("Loaded high correlation pairs:", spearman_corr_pairs)
