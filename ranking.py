import pandas as pd
import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

absolute_path = os.path.dirname(__file__)

mastersheet = pd.read_csv(os.path.join(absolute_path, "data/MASTERSHEET.csv"))
mastersheet.replace(["*", " "], np.nan, inplace=True)

id_columns_transferred = ["dc", "st", "district", "state", "HDI"]
df_ids = mastersheet[id_columns_transferred].copy()

rankings_data = pd.read_csv(os.path.join(absolute_path, "data/k_means_results.csv"))

# identify relevant columns based on the pattern
label_cols = list(range(5, 181, 5))  # columns for labels (6, 11, 16, ..., 181)
cluster_hdi_cols = list(range(9, 185, 5))  # corresponding cluster HDI columns (10, 15, ..., 185)
hdi_col = "HDI"  # HDI 

# create a copy of the dataframe to store the updated cluster HDI values
rankings_df = rankings_data.copy()

# process each label column
for label_col, hdi_col_update in zip(label_cols, cluster_hdi_cols):
    label_col_name = rankings_df.columns[label_col]
    hdi_col_name = rankings_df.columns[hdi_col_update]
    
    # label column is numeric for proper grouping
    rankings_df[label_col_name] = pd.to_numeric(rankings_df[label_col_name], errors='coerce')
    
    # mapping of label to average HDI, ignoring NaN values
    label_to_hdi = rankings_df.groupby(label_col_name, dropna=True)[hdi_col].mean()
    
    # average HDI to the respective cluster HDI column
    rankings_df[hdi_col_name] = rankings_df[label_col_name].map(label_to_hdi)

# save the updated dataframe
rankings_df.to_csv(os.path.join(absolute_path, "data/k_means_results.csv"), index=False)

time.sleep(2.5)

# nornmalize: z-score, min/max

# define the starting points
n, k, j = 6, 8, 9
end_col = 185

# generate the column indices using the pattern (n+5, k+5, j+5)
columns_to_normalize = [185, 186, 187]
while max(n, k, j) <= end_col:
    if n <= end_col:
        columns_to_normalize.append(n)
        n += 5
    if k <= end_col:
        columns_to_normalize.append(k)
        k += 5
    if j <= end_col:
        columns_to_normalize.append(j)
        j += 5

print(columns_to_normalize)

zscore_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Z-score normalization
rankings_df.iloc[:, columns_to_normalize] = zscore_scaler.fit_transform(rankings_df.iloc[:, columns_to_normalize])

# Min-Max scaling
rankings_df.iloc[:, columns_to_normalize] = minmax_scaler.fit_transform(rankings_df.iloc[:, columns_to_normalize])

print("Updated DataFrame:")
print(rankings_df.head())  # Display the updated dataframe

rankings_df.to_csv(os.path.join(absolute_path, "data/k_means_results_normalized.csv"), index=False)

# individual_ideal_distance score

# select the last three columns
last_three_cols = rankings_df.iloc[:, -3:]
# lesser the distance, it's better, so transformed
x_transformed = 1 / (last_three_cols + 1)

scaler = MinMaxScaler()
scores = pd.DataFrame(scaler.fit_transform(x_transformed), columns=[col + "_score" for col in last_three_cols.columns])

rankings_df = pd.concat([rankings_df, scores], axis=1)
print(rankings_df.head())

# cluster ranking

columns_to_normalize = columns_to_normalize[3:]
columns_to_normalize = [columns_to_normalize[i:i+3] for i in range(0, len(columns_to_normalize), 3)]
print(columns_to_normalize)

# weights for the last 3 columns (centroid_dist, sq_dist, HDI)
weights = [0.3, 0.3, 0.4]  

cluster_ranks_df = pd.DataFrame()  

for i, group in enumerate(columns_to_normalize, start=1):
    print(f"Processing group {i}: {group}")

    first_col, second_col, last_col = group  

    rankings_df.iloc[:, group] = rankings_df.iloc[:, group].apply(pd.to_numeric, errors="coerce")

    # transformation: x = 1 / (1 + cell value)
    x_values = rankings_df.iloc[:, [first_col, second_col]].apply(lambda x: 1 / (1 + x))

    # Min-Max scaling
    scaler = MinMaxScaler()
    x_values = pd.DataFrame(scaler.fit_transform(x_values), columns=["x1", "x2"])  # Renaming to avoid index confusion

    # HDI column’s values separately
    x_values["x3"] = rankings_df.iloc[:, last_col].values  

    # weighted score
    x_values["weighted_score"] = (
        x_values["x1"] * weights[0] +
        x_values["x2"] * weights[1] +
        x_values["x3"] * weights[2]
    )

    # append weighted column to final dataframe
    cluster_ranks_df[f"{i}_cluster_scores"] = x_values["weighted_score"]

cluster_ranks_df.to_csv(os.path.join(absolute_path, "data/cluster_rankings.csv"), index=False)

# assign weights
alpha, beta = 0.5, 0.5

result_df = pd.DataFrame(index=cluster_ranks_df.index)

# number of cols related to one type of underlying data
iters_per_score_col = 12

# iterate over cluster_ranks_df columns in chunks of `iters_per_score_col`
for i, col in enumerate(cluster_ranks_df.columns):
    score_col = scores.columns[i // iters_per_score_col % len(scores.columns)]  # Cycle through score columns
    result_df[col] = alpha * cluster_ranks_df[col] + beta * scores[score_col]

print(result_df)

final_df = df_ids.join(result_df)

# rank & rename each col
# process columns starting from index 6
for i in range(5, len(final_df.columns)):
    col_name = final_df.columns[i]
    
    # rename the column
    new_col_name = f"{i-4}_socioeconomic_scores"
    final_df.rename(columns={col_name: new_col_name}, inplace=True)
    
    # create the ranking column (descending order)
    rank_col_name = f"{i-4}_socioeconomic_ranks"
    final_df[rank_col_name] = final_df[new_col_name].rank(ascending=False, method='dense')

print(final_df)
final_df.to_csv(os.path.join(absolute_path, "data/results.csv"), index=False)


# compute and visualize Spearman correlation matrix
corr_matrix = final_df.iloc[:, -36:].corr(method='spearman').abs()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation Matrix")
plt.show()

# select upper triangle to avoid duplicate values
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# feature pairs with correlation >= 0.9
high_corr_pairs = []
for col in upper_tri.columns:
    high_corr = upper_tri[col][upper_tri[col] >= 0.9].index.tolist()
    for feature in high_corr:
        high_corr_pairs.append((col, feature))  # Store both features

# export
with open("data/spearman_cols.txt", "w") as file:
    file.writelines([f"{pair[0]},{pair[1]}\n" for pair in high_corr_pairs])

print(f"Spearman Column Pairs: {high_corr_pairs}")