import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import os

absolute_path = os.path.dirname(__file__)

hhv1 = pd.read_csv(os.path.join(absolute_path, "Data/PLFS/Data/hhv1.csv"))
perv1 = pd.read_csv(os.path.join(absolute_path, "Data/PLFS/Data/perv1.csv"))

print(hhv1.head(), perv1.head())

master_df = pd.read_excel(os.path.join(absolute_path, "data/NFHS_cleaned_with_HDI.xlsx"))

nss_mapping = pd.read_csv(os.path.join(absolute_path, "data/nss_codes.csv"))

# merge NSS codes with mastersheet based on both STATE and DISTRICT
master_df = master_df.merge(nss_mapping, on=["state", "district"], how="left")

print(master_df.head)

master_df.to_csv(os.path.join(absolute_path, "data/NFHS_cleaned_with_HDI_and_NSS_Codes.csv"), index=False)


# generate district level aggreagtes from hhv1

# no_qtr used because annual esitmates will be used
hhv1["weight"] = hhv1["mult"] / hhv1["no_qtr"] 

def proportion_distribution(column):
    return column.value_counts(normalize=True).to_dict()

# Function to get weighted mean
def weighted_mean(series, weights):
    return (series * weights).sum() / weights.sum()

# Define aggregation functions
agg_funcs = {
    "hh_size": ["mean", "median"],  
    "hce1": ["mean", "median"],  
    "hce2": ["mean", "median"],
    "hce3": ["mean", "median"],
    "hce4": ["mean", "median"],
    "hce5": ["mean", "median"],
    "hce_tot": ["mean", "median"],
    "relg": proportion_distribution,  
    "sg": proportion_distribution,  
    "mult": "sum",  # sum of multipliers for district
}

# group by state (`st`) & district (`dc`) using `mult`
district_agg = (
    hhv1.groupby(["st", "dc"])
    .apply(lambda g: pd.Series({
        "hh_size_mean": weighted_mean(g["hh_size"], g["mult"]),
        "hh_size_median": g["hh_size"].median(),
        "hce1_mean": weighted_mean(g["hce1"], g["mult"]),
        "hce1_median": g["hce1"].median(),
        "hce2_mean": weighted_mean(g["hce2"], g["mult"]),
        "hce2_median": g["hce2"].median(),
        "hce3_mean": weighted_mean(g["hce3"], g["mult"]),
        "hce3_median": g["hce3"].median(),
        "hce4_mean": weighted_mean(g["hce4"], g["mult"]),
        "hce4_median": g["hce4"].median(),
        "hce5_mean": weighted_mean(g["hce5"], g["mult"]),
        "hce5_median": g["hce5"].median(),
        "hce_tot_mean": weighted_mean(g["hce_tot"], g["mult"]),
        "hce_tot_median": g["hce_tot"].median(),
        "mult_total": g["mult"].sum(),  # Sum of multipliers
        "relg_dist": proportion_distribution(g["relg"]),
        "sg_dist": proportion_distribution(g["sg"]),
    }))
    .reset_index()
)
# expand religion & social group distributions 
relg_df = district_agg["relg_dist"].apply(pd.Series).fillna(0)
relg_df.columns = [f"relg_{int(col)}_pct" for col in relg_df.columns]  # Rename columns

sg_df = district_agg["sg_dist"].apply(pd.Series).fillna(0)
sg_df.columns = [f"sg_{int(col)}_pct" for col in sg_df.columns]  # Rename columns

district_agg = pd.concat([district_agg.drop(columns=["relg_dist", "sg_dist"]), relg_df, sg_df], axis=1)

print(district_agg.head())

district_agg.to_csv("data/PLFS_hhv1.csv", index=False)


# generate district level aggreagtes from perv1
def weighted_mean(series, weights):
    total_weight = weights.sum()
    return (series * weights).sum() / total_weight if total_weight > 0 else None

def weighted_categorical_distribution(series, weights, var_prefix):
    mask = series.notna()
    total = weights[mask].sum()
    if total == 0:
        return {}
    dist = weights[mask].groupby(series[mask]).sum() / total
    return {f"{var_prefix}_{cat}": dist_val for cat, dist_val in dist.items()}

def aggregate_perv1_data(perv1):
    # calculate weight 
    perv1["weight"] = perv1["mult"] / perv1["no_qtr"]
    
    # create age groups
    age_bins = [0, 18, 35, 60, float("inf")]
    age_labels = ["age_<18", "age_18-35", "age_35-60", "age_60+"]
    perv1["age_group"] = pd.cut(perv1["age"], bins=age_bins, labels=age_labels, right=False)

    # variables for aggregation
    continuous_vars = ["form_edu", "ern_reg", "ern_self"]
    cat_vars = [
        "age_group", "marst", "gedu_lvl", "tedu_lvl", "voc", "voc_fld", "voc_dur", "voc_typ",
        "voc_fund", "has_sas", "loc_pas", "etyp_pas", "wrkr_pas", "job_pas", "leave_pas",
        "ssec_pas", "ecoprd_pas", "loc_sas", "etyp_sas", "wrkr_sas", "job_sas", "leave_sas",
        "ssec_sas", "ecoprd_sas", "wrk_365", "dur_pas", "dur_sas", "eff_pas", "dur_unp",
        "evr_wrk", "acws"
    ]

    # district-level aggregates
    result_list = []
    for (st, dc), grp in perv1.groupby(["st", "dc"]):
        row = {"st": st, "dc": dc}
        
        for var in continuous_vars:
            row[f"{var}_mean"] = weighted_mean(grp[var], grp["weight"])
            row[f"{var}_median"] = grp[var].median()
        
        for var in cat_vars:
            dist_dict = weighted_categorical_distribution(grp[var], grp["weight"], var)
            row.update(dist_dict)
        
        result_list.append(row)

    return pd.DataFrame(result_list)

perv1_agg = aggregate_perv1_data(perv1)
print(perv1_agg.head())

perv1_agg.to_csv("data/PLFS_perv1.csv", index=False)


# merge all three datasets and drop cols with <67% data

def preprocess_df(df):
    # variable should cover at least 67% districts
    threshold = len(df) * 0.33
    
    # function to check if a column should be dropped
    def should_drop(col):
        return (
            (col.isin(['*', '', 0]).sum() > threshold) or
            (col.isna().sum() > threshold)
        )
    
    columns_to_drop = [col for col in df.columns if should_drop(df[col])]
    
    cleaned_df = df.drop(columns=columns_to_drop)
    
    return cleaned_df

df1 = pd.read_csv((os.path.join(absolute_path,'data/PLFS_perv1.csv')))
df2 = pd.read_csv((os.path.join(absolute_path,'data/PLFS_hhv1.csv')))
df3 = pd.read_csv((os.path.join(absolute_path,'data/NFHS_cleaned_with_HDI_and_NSS_Codes.csv')))

df1_cleaned = preprocess_df(df1)
df2_cleaned = preprocess_df(df2)
df3_cleaned = preprocess_df(df3)

print(df1_cleaned.head)
print(df2_cleaned.head)
print(df3_cleaned.head)

# merged current mastersheet with PLFS datasets
merged_3_1_2 = pd.merge(df3_cleaned, df1_cleaned, on=['st', 'dc'], how='inner')
merged_3_1_2 = pd.merge(merged_3_1_2, df2_cleaned, on=['st', 'dc'], how='inner')

merged_3_1_2.to_csv((os.path.join(absolute_path,'data/MASTERSHEET.csv')), index=False)

print("Preprocessing Completed... \n Saved @ 'MASTERSHEET.csv'.")