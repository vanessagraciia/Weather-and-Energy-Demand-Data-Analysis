import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# TRAINING
melb_df = pd.read_csv("melb_weather_normalized.csv")
melb_pdd_df = pd.read_csv("pdd_melb_normalized.csv")
melb = pd.merge(melb_df, melb_pdd_df, how='inner')
melb.set_index("Date", drop=True, inplace=True)

features = ['Mean Temperature (°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Speed of maximum wind gust (km/h)', 'TOTAL DEMAND']
melb = melb[features]

features_dict = {"Mean Temperature (°C)": 10, "Rainfall (mm)": 50, "Evaporation (mm)": 15, "Speed of maximum wind gust (km/h)": 15, "TOTAL DEMAND": 12}
for feature in features:
    equal_width = KBinsDiscretizer(n_bins=features_dict[feature], encode='ordinal', strategy='uniform')
    melb["Binned " + feature] = equal_width.fit_transform(melb[feature].to_numpy().reshape(-1, 1)).astype(int)

melb.to_csv("NEW_BINNED_AND_FILTERED_DATA.csv")

# TESTING
melb_df = pd.read_csv("melb_weather_normalized_testing.csv")
melb_pdd_df = pd.read_csv("pdd_melb_normalized_testing.csv")
melb = pd.merge(melb_df, melb_pdd_df, how='inner')
melb.set_index("Date", drop=True, inplace=True)

features = ['Mean Temperature (°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Speed of maximum wind gust (km/h)', 'TOTAL DEMAND']
melb = melb[features]

features_dict = {"Mean Temperature (°C)": 10, "Rainfall (mm)": 50, "Evaporation (mm)": 15, "Speed of maximum wind gust (km/h)": 15, "TOTAL DEMAND": 12}
for feature in features:
    equal_width = KBinsDiscretizer(n_bins=features_dict[feature], encode='ordinal', strategy='uniform')
    melb["Binned " + feature] = equal_width.fit_transform(melb[feature].to_numpy().reshape(-1, 1)).astype(int)
melb.to_csv("NEW_BINNED_AND_FILTERED_DATA_TESTING.csv")

