import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd

absolute_path = os.path.dirname(__file__)
results_df = pd.read_csv(os.path.join(absolute_path, "data/results.csv"))
india_shapefile = gpd.read_file(os.path.join(absolute_path, "shapefile/2011_Dist.shp"))

# merge on state and district codes
merged_df = india_shapefile.merge(results_df, left_on=["ST_CEN_CD", "DT_CEN_CD"], right_on=["st", "dc"])
merged_df = merged_df.drop(columns=["DISTRICT", "ST_NM"])

# INSERT DESIRED COLUMN
score_col = "12_socioeconomic_scores"  

# district-level ranking
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

merged_df.plot(
    column=score_col, cmap="coolwarm", linewidth=0.5, edgecolor="black",
    legend=True, ax=ax,
    legend_kwds={"shrink": 0.4},  
    missing_kwds={"color": "white", "label": "Missing Data"} 
)

india_shapefile.boundary.plot(ax=ax, color="black", linewidth=0.5)
ax.set_title(f"District-level Socio-economic Scores in India for {score_col}", fontsize=12, fontname="georgia")
ax.axis("off")

# Save as PNG
plt.savefig(f"output/maps/{score_col}_map.png", dpi=300, bbox_inches="tight")
plt.show()


# state-level ranking

# group by state code, calculate mean of score_col
state_avg_df = merged_df.groupby("ST_CEN_CD")[score_col].mean().reset_index()

# extract state names and merge
state_names = india_shapefile[["ST_CEN_CD", "ST_NM"]].drop_duplicates()
state_avg_df = state_avg_df.merge(state_names, on="ST_CEN_CD", how="left")
state_avg_df.to_csv(f"output/rank_statistics/{score_col}_state_avg.csv", index=False)
print(state_avg_df.head())

# merge with shapefile for state-level maps
state_merged_df = india_shapefile.dissolve(by="ST_CEN_CD").reset_index()
state_merged_df = state_merged_df.merge(state_avg_df, on="ST_CEN_CD", how="left")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

state_merged_df.plot(
    column=score_col, cmap="coolwarm",
    linewidth=0, edgecolor="none",  # remove district boundaries
    legend=True, ax=ax,
    legend_kwds={"shrink": 0.4},
    missing_kwds={"color": "white", "label": "Missing Data"}  
)

# state boundaries
india_shapefile.dissolve(by="ST_CEN_CD").boundary.plot(ax=ax, color="black", linewidth=0.8)

ax.set_title(f"State-level Socio-economic Scores in India for {score_col}", fontsize=12, fontname="georgia")
ax.axis("off")

plt.savefig(f"output/maps/{score_col}_state_map.png", dpi=300, bbox_inches="tight")
plt.show()
