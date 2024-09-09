'''Preprocessing'''
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import datetime

"""
MELBOURNE
"""
#PRE-PROCESS MELBOURNE WEATHER DATASET
melb_df = pd.read_csv("testing_melb_weather.csv")

# Filter data: Only interested in quantitative data
melb_weather = melb_df[["Date", "Minimum temperature (°C)", "Maximum temperature (°C)", "3pm Temperature (°C)", "9am Temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
"Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]

# Missing Values: Impute Missing Values with Mean or interpolate?
melb_imputed = melb_weather[["Minimum temperature (°C)", "Maximum temperature (°C)",  "3pm Temperature (°C)", "9am Temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
"Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]
melb_imputed = melb_imputed.fillna(melb_imputed.mean())

# Find Mean Value and Insert to Dataframe
mean_temp = []
for i in range(len(melb_imputed)):
    mean_temp.append((melb_imputed["Minimum temperature (°C)"][i] + melb_imputed["Maximum temperature (°C)"][i])/2)
    # mean_temp.append((melb_imputed["Minimum temperature (°C)"][i] + melb_imputed["Maximum temperature (°C)"][i] + melb_imputed["3pm Temperature (°C)"][i] + melb_imputed["9am Temperature (°C)"][i])/4)
melb_imputed.insert(2, "Mean Temperature (°C)", mean_temp)

# Include date 
melb_imputed.insert(0, "Date", melb_weather["Date"])
melb_imputed.set_index("Date", drop=True, inplace=True)

# Output pre-processed Melbourne Weather Dataset as csv
melb_imputed.to_csv("melb_weather_testing.csv")

# Normalize: Rescale the values to be between 0 and 1
melb_normalized = (melb_imputed - melb_imputed.min()) / (melb_imputed.max() - melb_imputed.min())

# DELETE KALO ERROR
# Output pre-processed normalized Melbourne Weather Dataset as csv
melb_normalized.to_csv("melb_weather_normalized_testing.csv")

#GET AND NORMALIZE PRICE DEMAND DATASET
pdd_melb = melb_df[["Date", "TOTAL DEMAND"]]

# Output pre-processed Melbourne Price Demand Dataset as csv
pdd_melb.to_csv("new_pdd_melb_testing.csv")

# Normalize: Rescale the values to be between 0 and 1
pdd_melb_normalized = (pdd_melb["TOTAL DEMAND"] - pdd_melb["TOTAL DEMAND"].min()) / (pdd_melb["TOTAL DEMAND"].max() - pdd_melb["TOTAL DEMAND"].min())
pdd_melb.insert(1, "Normalized", pdd_melb_normalized)
pdd_melb.to_csv("pdd_melb_normalized_testing.csv")
