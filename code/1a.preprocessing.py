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
melb_df = pd.read_csv("training_melb_weather.csv")

# Filter data: Only interested in Quantitative Data
melb_weather = melb_df[["Date", "Minimum temperature (°C)", "Maximum temperature (°C)", "3pm Temperature (°C)", "9am Temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
"Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]

# Missing Values: Impute Missing Values with Mean or interpolate?
melb_imputed = melb_weather[["Minimum temperature (°C)", "Maximum temperature (°C)",  "3pm Temperature (°C)", "9am Temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
"Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]
melb_imputed = melb_imputed.fillna(melb_imputed.mean())

# Find Mean Temperature Value and Insert to Dataframe
mean_temp = []
for i in range(len(melb_imputed)):
    mean_temp.append((melb_imputed["Minimum temperature (°C)"][i] + melb_imputed["Maximum temperature (°C)"][i])/2)
melb_imputed.insert(2, "Mean Temperature (°C)", mean_temp)

# Include date 
melb_imputed.insert(0, "Date", melb_weather["Date"])
melb_imputed.set_index("Date", drop=True, inplace=True)

# Output pre-processed Melbourne Weather Dataset as csv
melb_imputed.to_csv("melb_weather.csv")

# Normalize: Rescale the values to be between 0 and 1
melb_normalized = (melb_imputed - melb_imputed.min()) / (melb_imputed.max() - melb_imputed.min())

# Output pre-processed normalized Melbourne Weather Dataset as csv
melb_normalized.to_csv("melb_weather_normalized.csv") 

#GET AND NORMALIZE PRICE DEMAND DATASET
pdd_melb = melb_df[["Date", "TOTAL DEMAND"]]

# Output pre-processed Melbourne Price Demand Dataset as csv
pdd_melb.to_csv("new_pdd_melb.csv")

# Normalize: Rescale the values to be between 0 and 1
pdd_melb_normalized = (pdd_melb["TOTAL DEMAND"] - pdd_melb["TOTAL DEMAND"].min()) / (pdd_melb["TOTAL DEMAND"].max() - pdd_melb["TOTAL DEMAND"].min())
pdd_melb.insert(1, "Normalized", pdd_melb_normalized)
pdd_melb.to_csv("pdd_melb_normalized.csv")

pdd_melb.set_index("Date")
# """
# SYDNEY
# """ #PRE-PROCESS SYDNEY WEATHER DATASET
# syd_df = pd.read_csv("weather_sydney.csv")

# # Filter data: Only interested in quantitative data
# syd_weather = syd_df[["Date", "Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]

# # Missing Values: Impute Missing Values with Mean or interpolate?
# syd_imputed = syd_df[["Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]
# syd_imputed = syd_imputed.fillna(syd_imputed.median())

# # Find Mean Value and Insert to Dataframe
# mean_temp = []
# for i in range(len(syd_imputed)):
#     mean_temp.append((syd_imputed["Minimum temperature (°C)"][i] + syd_imputed["Maximum temperature (°C)"][i])/2)
# syd_imputed.insert(2, "Mean Temperature (°C)", mean_temp)

# # Change Date Format, Insert Date to DataFrame
# date = syd_weather["Date"]
# new_date_format = []
# for value in date:
#     value = datetime.datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y/%m/%d")
#     new_date_format.append(value)
# syd_imputed.insert(0, "Date", new_date_format)
# syd_imputed.set_index("Date", inplace=True)

# # Output pre-processed Sydney Weather Dataset as csv
# syd_imputed.to_csv("syd_weather.csv")

# # Normalize: Rescale the values to be between 0 and 1
# syd_normalized = (syd_imputed - syd_imputed.min()) / (syd_imputed.max() - syd_imputed.min())

# # Output pre-processed normalized Sydney Weather Dataset as csv
# syd_normalized.to_csv("syd_weather_normalized.csv") 

# #PRE-PROCESS PRICE DEMAND DATASET
# pdd_df = pd.read_csv("price_demand_data.csv")

# # Get dataframe for NSW
# pdd_syd = pdd_df[pdd_df["REGION"] == "NSW1"]

# # Change date format to daily date
# date_list = []
# for date in pdd_syd["SETTLEMENTDATE"]:
#     date_list.append(date[:-9])
# pdd_syd.insert(1, "Date", date_list)

# # Find sum of energy for each day
# pdd_syd = round(pdd_syd.groupby("Date")["TOTALDEMAND"].sum(), 2)
# pdd_syd.rename("TOTAL DEMAND", inplace=True)
# pdd_syd = pdd_syd.drop(pdd_syd.index[-1])

# # Output pre-processed Sydney Price Demand Dataset as csv
# pdd_syd.to_csv("pdd_syd.csv")




# """
# ADELAIDE
# """
# #PRE-PROCESS ADELAIDE WEATHER DATASET
# adld_df = pd.read_csv("weather_adelaide.csv")

# # Filter data: Only interested in quantitative data
# adld_weather = adld_df[["Date", "Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]

# # Missing Values: Impute Missing Values with Mean or interpolate?
# adld_imputed = adld_df[["Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]
# adld_imputed = adld_imputed.fillna(adld_imputed.median())

# # Find Mean Value and Insert to Dataframe
# mean_temp = []
# for i in range(len(adld_imputed)):
#     mean_temp.append((adld_imputed["Minimum temperature (°C)"][i] + adld_imputed["Maximum temperature (°C)"][i])/2)
# adld_imputed.insert(2, "Mean Temperature (°C)", mean_temp)

# # Change Date Format, Insert Date to DataFrame
# date = adld_weather["Date"]
# new_date_format = []
# for value in date:
#     value = datetime.datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y/%m/%d")
#     new_date_format.append(value)
# adld_imputed.insert(0, "Date", new_date_format)
# adld_imputed.set_index("Date", inplace=True)

# # Output pre-processed Adelaide Weather Dataset as csv
# adld_imputed.to_csv("adld_weather.csv")

# # Normalize: Rescale the values to be between 0 and 1
# adld_normalized = (adld_imputed - adld_imputed.min()) / (adld_imputed.max() - adld_imputed.min())

# # Output pre-processed normalized Adelaide Weather Dataset as csv
# adld_normalized.to_csv("adld_weather_normalized.csv") 

# #PRE-PROCESS PRICE DEMAND DATASET
# pdd_df = pd.read_csv("price_demand_data.csv")

# # Get dataframe for SA
# pdd_adld = pdd_df[pdd_df["REGION"] == "SA1"]

# # Change date format to daily date
# date_list = []
# for date in pdd_adld["SETTLEMENTDATE"]:
#     date_list.append(date[:-9])
# pdd_adld.insert(1, "Date", date_list)

# # Find sum of energy for each day
# pdd_adld = round(pdd_adld.groupby("Date")["TOTALDEMAND"].sum(), 2)
# pdd_adld.rename("TOTAL DEMAND", inplace=True)
# pdd_adld = pdd_adld.drop(pdd_adld.index[-1])

# # Output pre-processed Adelaide Price Demand Dataset as csv
# pdd_adld.to_csv("pdd_adld.csv")




# """
# BRISBANE
# """

# #PRE-PROCESS BRISBANE WEATHER DATASET
# bris_df = pd.read_csv("weather_brisbane.csv")

# # Filter data: Only interested in quantitative data
# bris_weather = bris_df[["Date", "Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]

# # Missing Values: Impute Missing Values with Mean or interpolate?
# bris_imputed = bris_df[["Minimum temperature (°C)", "Maximum temperature (°C)", "Rainfall (mm)", "Evaporation (mm)", 
# "Sunshine (hours)", "Speed of maximum wind gust (km/h)"]]
# bris_imputed = bris_imputed.fillna(bris_imputed.median())

# # Find Mean Value and Insert to Dataframe
# mean_temp = []
# for i in range(len(bris_imputed)):
#     mean_temp.append((bris_imputed["Minimum temperature (°C)"][i] + bris_imputed["Maximum temperature (°C)"][i])/2)
# bris_imputed.insert(2, "Mean Temperature (°C)", mean_temp)

# # Include and Set Date as Index
# date = bris_weather["Date"]
# new_date_format = []
# for value in date:
#     value = datetime.datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y/%m/%d")
#     new_date_format.append(value)
# bris_imputed.insert(0, "Date", new_date_format)
# bris_imputed.set_index("Date", inplace=True)

# # Output pre-processed Brisbane Weather Dataset as csv
# bris_imputed.to_csv("bris_weather.csv")

# # Normalize: Rescale the values to be between 0 and 1
# bris_normalized = (bris_imputed - bris_imputed.min()) / (bris_imputed.max() - bris_imputed.min())

# # Output pre-processed normalized Brisbane Weather Dataset as csv
# bris_normalized.to_csv("bris_weather_normalized.csv") 

# #PRE-PROCESS PRICE DEMAND DATASET
# pdd_df = pd.read_csv("price_demand_data.csv")

# # Get dataframe for QLD
# pdd_bris = pdd_df[pdd_df["REGION"] == "QLD1"]

# # Change date format to daily date
# date_list = []
# for date in pdd_bris["SETTLEMENTDATE"]:
#     date_list.append(date[:-9])
# pdd_bris.insert(1, "Date", date_list)

# # Find sum of energy for each day
# pdd_bris = round(pdd_bris.groupby("Date")["TOTALDEMAND"].sum(), 2)
# pdd_bris.rename("TOTAL DEMAND", inplace=True)
# pdd_bris = pdd_bris.drop(pdd_bris.index[-1])

# # Output pre-processed Brisbane Price Demand Dataset as csv
# pdd_bris.to_csv("pdd_bris.csv")

