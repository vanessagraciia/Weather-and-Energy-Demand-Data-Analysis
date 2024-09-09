 import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif

#Get General Statistics (From Non-Normalized Dataset)
melb_nn_df = pd.read_csv("melb_weather.csv")
melb_pdd_nn_df = pd.read_csv("pdd_melb.csv")
melb_nn = pd.merge(melb_nn_df, melb_pdd_nn_df, how='inner')
stats = melb_nn.describe()

melb_df = pd.read_csv("melb_weather_normalized.csv")
melb_pdd_df = pd.read_csv("pdd_melb_normalized.csv")
melb = pd.merge(melb_df, melb_pdd_df, how='inner')
melb.set_index("Date", drop=True, inplace=True)
melb.drop("Unnamed: 0", inplace=True, axis=1)
melb.drop("Normalized", inplace=True, axis=1)

# NOTE: NOT LINEAR
# Find Correlation Between Each Weather Component and Total Demand
corr_data = melb.corr(method='spearman')["TOTAL DEMAND"]
print(corr_data)

# FEATURE SELECTION BASED ON MUTUAL INFORMATION

# Discretize continuous data
for feature in melb_nn.iloc[:,1:]:
    q3, q1 = np.percentile(melb_nn[feature], [75, 25])
    iqr = q3 - q1
    h = 2 * iqr / len(melb)**(1/3)
    bins = int((melb_nn[feature].max() - melb_nn[feature].min()) / h) # Find number of bins
    print(feature, bins)
    equal_width = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    melb["Binned " + feature] = equal_width.fit_transform(melb[feature].to_numpy().reshape(-1, 1)).astype(int)

# binned_melb = melb[["Binned Minimum temperature (°C)", "Binned Maximum temperature (°C)", "Binned Mean Temperature (°C)", "Binned Rainfall (mm)", "Binned Evaporation (mm)", "Binned Sunshine (hours)", "Binned Speed of maximum wind gust (km/h)", "Binned TOTAL DEMAND"]]
# binned_melb.to_csv("filtered_melb_weather.csv")

#Calculate mutual information
filtered_features = []
THRESHOLD = 0.2 
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
filtered_melb.to_csv("filtered_melb_weather.csv")

# visualize histogram of demand against frequency of bins
plt.hist(melb["Binned TOTAL DEMAND"])
plt.xlabel("Total Demand")
plt.ylabel("Frequency")
plt.title("Victoria")
plt.savefig("melb_demand_freq.png")
plt.close()

