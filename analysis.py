import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

absolute_path = os.path.dirname(__file__)
mastersheet = pd.read_csv(os.path.join(absolute_path, "data/MASTERSHEET.csv"))

mastersheet.replace(["*", " "], np.nan, inplace=True)

# identifier columns
id_columns = ["dc", "st", "district", "state", "HDI", "mult_total"]
id_columns_transferred = ["dc", "st", "district", "state", "HDI"]
df_ids = mastersheet[id_columns_transferred].copy()

# new DataFrame for analysis by dropping the ID columns
df_analysis = mastersheet.drop(columns=id_columns)

# add column showing missing variables (count of)
df_analysis["missing_count"] = df_analysis.isna().sum(axis=1)

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
# df_analysis.to_csv(os.path.join(absolute_path, "data/MASTERSHEET_after_Spearman.csv"), index=False)

# Z-score normalization
scaler = StandardScaler()
df_analysis_zscore = pd.DataFrame(scaler.fit_transform(df_analysis), columns=df_analysis.columns)

# handle missing values -> missing variable column & KNN
# weights="distance" 
# assumption -> if 2 districts are similar across the observed features, they have similar values for the missing features 
knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
df_analysis_zscore.iloc[:, :-1] = knn_imputer.fit_transform(df_analysis_zscore.drop(columns=["missing_count"]))

print(df_analysis_zscore.head())
df_analysis_zscore.to_csv(os.path.join(absolute_path, "data/normalized_df_analysis_imputed.csv"), index=False)

# Pearson correlation
pearson_corr = df_analysis_zscore.corr(method='pearson')
pearson_corr.to_csv("data/pearson_correlation_matrix.csv")

# extract off-diagonal elements
off_diag_values = pearson_corr.where(~np.eye(pearson_corr.shape[0], dtype=bool)).stack()
total_off_diag = len(off_diag_values) 

# compute descriptive statistics
off_diag_summary = {
    "mean": off_diag_values.mean(),
    "median": off_diag_values.median(),
    "std_dev": off_diag_values.std(),
    "min": off_diag_values.min(),
    "max": off_diag_values.max(),
    "prop_corr_below_25_%": ((off_diag_values.abs() < 0.25).sum() / total_off_diag) * 100,
    "prop_corr_above_50_%": ((off_diag_values.abs() > 0.50).sum() / total_off_diag) * 100,  
    "prop_corr_above_75_%": ((off_diag_values.abs() > 0.75).sum() / total_off_diag) * 100,  
    "prop_corr_above_90_%": ((off_diag_values.abs() > 0.90).sum() / total_off_diag) * 100, 
}

off_diag_summary_df = pd.DataFrame([off_diag_summary])
print(off_diag_summary_df.head(n=10))

# PCA
df_vanilla = df_analysis_zscore.copy()  
df_pca_3 = df_analysis_zscore.copy()    # PCA with 3 components
df_pca_opt = df_analysis_zscore.copy()  # PCA with explained variance 

pca_full = PCA()
pca_full.fit(df_analysis_zscore)

# explained variance and elbow plot
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o', linestyle='--', label="Cumulative Variance")
plt.axhline(y=95, color='r', linestyle='--', label="95% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Scree Plot: Selecting Optimal PCA Components")
plt.legend()
plt.grid(True)
plt.savefig("output/scree_plot.png")  # Export plot
plt.close()

# optimal components for 80% variance
optimal_components = np.argmax(cumulative_variance >= 0.8) + 1

# PCA with 3 and optimal components
pca_3 = PCA(n_components=3)
df_pca_3_transformed = pca_3.fit_transform(df_pca_3)

pca_optimal = PCA(n_components=optimal_components)
df_pca_opt_transformed = pca_optimal.fit_transform(df_pca_opt)

print(f"Optimal Number of PCA Components (80% Variance): {optimal_components}")

# feature weights in each PCA component
pca_3_weights = pd.DataFrame(np.abs(pca_3.components_), columns=df_analysis_zscore.columns)
pca_opt_weights = pd.DataFrame(np.abs(pca_optimal.components_), columns=df_analysis_zscore.columns)

# export top n features for each PCA method
with open("output/top_features_pca_3.txt", "w") as f:
    f.write("Top Features in PCA (3):\n")
    f.write(pca_3_weights.sum(axis=0).sort_values(ascending=False).head(15).to_string())

with open("output/top_features_pca_opt.txt", "w") as f:
    f.write("Top Features in PCA (Optimal Components for 80% Variance):\n")
    f.write(pca_opt_weights.sum(axis=0).sort_values(ascending=False).head(15).to_string())

# convert PCA-transformed data 
df_pca_3 = pd.DataFrame(df_pca_3_transformed, columns=[f"PC{i+1}" for i in range(3)])
df_pca_opt = pd.DataFrame(df_pca_opt_transformed, columns=[f"PC{i+1}" for i in range(optimal_components)])

# apply Min-Max scaling
scaler = MinMaxScaler()
df_vanilla_scaled = pd.DataFrame(scaler.fit_transform(df_vanilla), columns=df_vanilla.columns)
df_pca_3_scaled = pd.DataFrame(scaler.fit_transform(df_pca_3), columns=df_pca_3.columns)
df_pca_opt_scaled = pd.DataFrame(scaler.fit_transform(df_pca_opt), columns=df_pca_opt.columns)

# compute Euclidean distances from ideal district
def compute_distances(df, name):
    last_row = df.iloc[-1].values
    distances = np.linalg.norm(df.values - last_row, axis=1)
    return pd.DataFrame(distances, columns=[f'ind_dist_{name}'])

dist_vanilla = compute_distances(df_vanilla_scaled, "vanilla")
dist_pca_3 = compute_distances(df_pca_3_scaled, "pca_3")
dist_pca_opt = compute_distances(df_pca_opt_scaled, "pca_opt")

df_distances = pd.concat([dist_vanilla, dist_pca_3, dist_pca_opt], axis=1)

output_path = "data/ind_dist_ideal.csv"
df_distances.to_csv(output_path, index=False)

# remove last row, store as ideal row, and combine the datasets
df_vanilla_scaled.iloc[[-1]].to_csv("data/ideal_district/vanilla.csv", index=False)
df_pca_3_scaled.iloc[[-1]].to_csv("data/ideal_district/pca_3.csv", index=False)
df_pca_opt_scaled.iloc[[-1]].to_csv("data/ideal_district/pca_opt.csv", index=False)

df_vanilla_scaled = df_vanilla_scaled.iloc[:-1]
df_pca_3_scaled = df_pca_3_scaled.iloc[:-1]
df_pca_opt_scaled = df_pca_opt_scaled.iloc[:-1]

print("\nProcessing Complete! Files saved:")
print("- Scree Plot: output/scree_plot.png")
print("- Ideal Row: data/ideal_row.csv")

# train-test splits [33/67, 67/33, 100/0]

post_pca_datasets = [df_vanilla_scaled, df_pca_3_scaled, df_pca_opt_scaled]
dataset_names = ["vanilla", "pca_3", "pca_opt"]

splits = [(0.33, 0.67), (0.67, 0.33), (1.0, 0.0)]  

# store train-test lists for each dataset
pre_k_means_datasets = []

for dataset, name in zip(post_pca_datasets, dataset_names):
    dataset_splits = []  # store splits for this dataset
    
    for train_size, test_size in splits:
        if test_size > 0:
            X_train, X_test = train_test_split(dataset, train_size=train_size, test_size=test_size, random_state=42)
            dataset_splits.extend([X_train, X_test])  # store both train & test
        else:
            X_train = dataset  # 100% train, no test set
            dataset_splits.append(X_train)  

    pre_k_means_datasets.append(dataset_splits)  # store for this dataset

# `pre_k_means_datasets` contains:
# pre_k_means_datasets[0] -> [vanilla_train_33_67, vanilla_test_33_67, vanilla_train_67_33, vanilla_test_67_33, vanilla_train_100_0]
# pre_k_means_datasets[1] -> [pca_3_train_33_67, pca_3_test_33_67, pca_3_train_67_33, pca_3_test_67_33, pca_3_train_100_0]
# pre_k_means_datasets[2] -> [pca_opt_train_33_67, pca_opt_test_33_67, pca_opt_train_67_33, pca_opt_test_67_33, pca_opt_train_100_0]

print("train-test splits stored in `pre_k_means_datasets` list")

# k-means 

# range of clusters to evaluate
k_values = range(2, 20) 

# store silhouette scores for reference
# only 100% train set used because of generalizabity
silhouette_scores = {}
for dataset, name in zip(pre_k_means_datasets, dataset_names):
    X_train_100_0 = dataset[-1] 
    
    inertia = []
    sil_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_train_100_0)

        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_train_100_0, labels))

    # store silhouette scores for reference
    silhouette_scores[name] = dict(zip(k_values, sil_scores))

    # plot Elbow diagram (Inertia)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, marker='o', linestyle='--', label="Inertia (WCSS)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title(f"Elbow Method for {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/{name}_elbow.png")
    plt.close()

    # plot Silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sil_scores, marker='o', linestyle='--', color='green', label="Silhouette Score")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Scores for {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/{name}_silhouette.png")
    plt.close()

    print(f"✅ Elbow & Silhouette plots saved for {name} at output/{name}_elbow.png & output/{name}_silhouette.png")

# SELECTED number of clusters based on Silhouette & Inertia -> n = 7,10 & n = 2,4 (for other reasons)
k_values = [2, 4, 7, 10]

# initialize DataFrame to store results
final_cluster_df = pd.DataFrame()

# iterate through each dataset 
for dataset_idx, dataset in enumerate(pre_k_means_datasets):
    dataset_name = dataset_names[dataset_idx]

    # ideal reference vector for this dataset
    ideal_vector = pd.read_csv(f"data/ideal_district/{dataset_name}.csv").values.flatten()
    
    train_33_67, test_33_67, train_67_33, test_67_33, train_100_0 = dataset

    # iterate over each K
    for k in k_values:
        for (train, test, split_name) in zip(
            [train_33_67, train_67_33, train_100_0],
            [test_33_67, test_67_33, None],  # no test set for 100_0
            ["33_67", "67_33", "100_0"]
        ):
            # fit KMeans on training data
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(train)

            # store train labels
            train_labels = kmeans.labels_
            train_centroids = kmeans.cluster_centers_

            # compute train centroid distances
            train_distances = np.array([euclidean(train_centroids[label], ideal_vector) for label in train_labels])

            # compute sum of squared distances for each cluster in training set
            train_sq_dist = np.zeros(len(train_labels))
            for cluster in range(k):
                cluster_distances = train_distances[train_labels == cluster] ** 2  # squaring distances
                sum_sq_dist = np.sum(cluster_distances)  # summing squared distances
                train_sq_dist[train_labels == cluster] = sum_sq_dist  # assign to each data point in that cluster

            # store train data 
            train_df = pd.DataFrame({
                f"{dataset_name}_{k}_{split_name}_label": train_labels,
                f"{dataset_name}_{k}_{split_name}_centroid_dist": train_distances,
                f"{dataset_name}_{k}_{split_name}_split": "train",
                f"{dataset_name}_{k}_{split_name}_sum_sq_dist": train_sq_dist,
                f"{dataset_name}_{k}_{split_name}_cluster_HDI": "",
            })

            # store test results if test set exists
            if test is not None:
                test_labels = kmeans.predict(test)
                test_distances = np.array([euclidean(train_centroids[label], ideal_vector) for label in test_labels])

                # compute sum of squared distances for each cluster in test set
                test_sq_dist = np.zeros(len(test_labels))
                for cluster in range(k):
                    cluster_distances = train_distances[train_labels == cluster] ** 2  # squaring distances
                    sum_sq_dist = np.sum(cluster_distances)  # summing squared distances
                    test_sq_dist[test_labels == cluster] = sum_sq_dist  # assign to each data point in that cluster

                test_df = pd.DataFrame({
                    f"{dataset_name}_{k}_{split_name}_label": test_labels,
                    f"{dataset_name}_{k}_{split_name}_centroid_dist": test_distances,
                    f"{dataset_name}_{k}_{split_name}_split": "test",
                    f"{dataset_name}_{k}_{split_name}_sum_sq_dist": test_sq_dist,
                    f"{dataset_name}_{k}_{split_name}_cluster_HDI": "",
                })

                # append test data
                train_df = pd.concat([train_df, test_df], ignore_index=True)

            # append results 
            final_cluster_df = pd.concat([final_cluster_df, train_df], axis=1)

# merge with df_ids (same row order)
final_cluster_df = df_ids.join(final_cluster_df)

print(final_cluster_df.head())

final_cluster_df = final_cluster_df.join(df_distances)

final_cluster_df.to_csv("data/k_means_results.csv", index=False)