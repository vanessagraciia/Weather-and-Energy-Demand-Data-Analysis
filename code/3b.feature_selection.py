import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif

# Get General Statistics (From Non-Normalized Dataset)
melb_nn_df = pd.read_csv("melb_weather_testing.csv")
melb_pdd_nn_df = pd.read_csv("new_pdd_melb_testing.csv")
melb_nn = pd.merge(melb_nn_df, melb_pdd_nn_df, how='inner')
stats = melb_nn.describe()
# print(melb_nn_df)
# print(melb_pdd_nn_df)
# print(melb_nn)
# FEATURE SELECTION ON MELBOURNE DATASET (From Normalized Dataset)
melb_df = pd.read_csv("melb_weather_normalized_testing.csv")
melb_pdd_df = pd.read_csv("pdd_melb_normalized_testing.csv")
melb = pd.merge(melb_df, melb_pdd_df, how='inner')
melb.set_index("Date", drop=True, inplace=True)


# FEATURE SELECTION BASED ON MUTUAL INFORMATION
binny_dict = {"Minimum temperature (°C)": 11, "Maximum temperature (°C)":12, "Mean Temperature (°C)": 10, "Rainfall (mm)": 240, "Evaporation (mm)":11,  "Sunshine (hours)": 7, "Speed of maximum wind gust (km/h)": 16, "TOTAL DEMAND": 12, '3pm Temperature (°C)': 10, '9am Temperature (°C)':10, 'Unnamed: 0':10}
# Discretize continuous data
for feature in melb_nn.iloc[:,1:]:
    q3, q1 = np.percentile(melb_nn[feature], [75, 25])
    iqr = q3 - q1
    h = 2 * iqr / len(melb)**(1/3)
    bins = binny_dict[feature]
    print(feature, bins)
    equal_width = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    melb["Binned " + feature] = equal_width.fit_transform(melb[feature].to_numpy().reshape(-1, 1)).astype(int)

# Calculate mutual information
filtered_features = []
THRESHOLD = 0.25
features = melb[["Binned Minimum temperature (°C)", "Binned Maximum temperature (°C)", "Binned Mean Temperature (°C)", "Binned Rainfall (mm)", "Binned Evaporation (mm)", "Binned Sunshine (hours)", "Binned Speed of maximum wind gust (km/h)", "Binned TOTAL DEMAND"]]
class_label = melb["Binned TOTAL DEMAND"]
mi_arr = mutual_info_classif(X=features, y=class_label, discrete_features=True)
for feature, mi in zip(features.columns, mi_arr):
    print(f'MI value for feature "{feature}": {mi:.4f}')
    if(mi >= THRESHOLD): 
        filtered_features.append(feature)
        
print('\nFeature set after filtering with MI:', filtered_features)

filtered_data = []
for feature in filtered_features:
    filtered_data.append(melb[feature])
filtered_melb = (pd.DataFrame(filtered_data)).T
demand = melb["Binned TOTAL DEMAND"]
filtered_melb.insert(0, "Binned_Total_Demand", demand)
filtered_melb.to_csv("filtered_melb_weather_testing.csv")

