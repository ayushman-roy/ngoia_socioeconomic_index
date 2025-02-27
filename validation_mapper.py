import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.cm as cm

absolute_path = os.path.dirname(__file__)

road_df = pd.read_csv(os.path.join(absolute_path, "validation/data/road_data.csv"))
population_df = pd.read_csv(os.path.join(absolute_path, "validation/data/pop_growth.csv"))

india_shapefile = gpd.read_file(os.path.join(absolute_path, "shapefile/2011_Dist.shp"))
nss_state = india_shapefile[["ST_CEN_CD", "ST_NM"]].drop_duplicates()


# merge road data with state names, shapefile
road_df = road_df.merge(nss_state, left_on="state", right_on="ST_NM", how="left")
road_df = road_df.drop(columns=["state", "ST_CEN_CD"])
road_df = india_shapefile.merge(road_df, on="ST_NM", how="left")

# plot road data
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

road_df.plot(
    column=road_df["road_len"], cmap="coolwarm",
    linewidth=0, edgecolor="none",  
    legend=True, ax=ax,
    legend_kwds={"shrink": 0.4},
    missing_kwds={"color": "white", "label": "Missing Data"}  
)

# state boundaries
india_shapefile.dissolve(by="ST_CEN_CD").boundary.plot(ax=ax, color="black", linewidth=0.8)

ax.set_title(f"State-level Distribution of District Roads (2019)", fontsize=12, fontname="georgia")
ax.axis("off")

plt.savefig(f"validation/output/road_network_state_map.png", dpi=300, bbox_inches="tight")
plt.show()


# merge population projected growth data with state names, shapefile
population_df = population_df.merge(nss_state, left_on="state", right_on="ST_NM", how="left")
population_df = population_df.drop(columns=["state", "ST_CEN_CD"])
population_df = india_shapefile.merge(population_df, on="ST_NM", how="left")

# ensure "10y_growth" is numeric
population_df["10y_growth"] = (
    population_df["10y_growth"]
    .astype(str)  
    .str.replace("%", "", regex=True) 
    .str.strip()  
    .replace("", float("nan"))
    .astype(float)
)

# plot population projected growth data
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

cmap = plt.cm.coolwarm
norm = mcolors.Normalize(vmin=population_df["10y_growth"].min(), vmax=population_df["10y_growth"].max())

population_df.plot(
    column="10y_growth", cmap=cmap, linewidth=0, edgecolor="black",
    legend=False, ax=ax, missing_kwds={"color": "white", "label": "Missing Data"}
)

cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
cbar.set_label("Projected Population Growth (%)", fontsize=10)

# state boundaries
india_shapefile.dissolve(by="ST_CEN_CD").boundary.plot(ax=ax, color="black", linewidth=0.8)

ax.set_title(f"State-level Projected Population Growth (2016-2026)", fontsize=12, fontname="georgia")
ax.axis("off")

plt.savefig(f"validation/output/population_growth_state_map.png", dpi=300, bbox_inches="tight")
plt.show()
