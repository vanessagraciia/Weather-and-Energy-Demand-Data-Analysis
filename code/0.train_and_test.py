from sklearn.model_selection import train_test_split
import pandas as pd
import datetime

# Data Linkage + Merge Dataset + Divide Data to Train and Test
melb_df = pd.read_csv("weather_melbourne.csv")
pdd_df = pd.read_csv("price_demand_data.csv")

# Change Date Format for Data Linkage
date = melb_df["Date"]
melb_df.drop(labels="Date", axis=1, inplace=True)
new_date_format = []
for value in date:
    value = datetime.datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y/%m/%d")
    new_date_format.append(value)
melb_df.insert(0, "Date", new_date_format)

# Filter out Melbourne Data
pdd_melb = pdd_df[pdd_df["REGION"] == "VIC1"]

# Change date format to daily date
date_list = []
for date in pdd_melb["SETTLEMENTDATE"]:
    date_list.append(date[:-9])
pdd_melb.insert(1, "Date", date_list)

# Find mean of energy for each day 
pdd_melb = round(pdd_melb.groupby("Date")["TOTALDEMAND"].mean(), 2)
pdd_melb.rename("TOTAL DEMAND", inplace=True)
# pdd_melb.set_index("Date")
pdd_melb.to_csv("pdd_melb.csv")

# Merge Data (Weather and Energy Demand)
pdd_df = pd.read_csv("pdd_melb.csv")
merged_data = pd.merge(melb_df, pdd_df, how='inner')

train_data, test_data = train_test_split(merged_data, test_size=0.2)

train_data.to_csv("training_melb_weather.csv")
test_data.to_csv("testing_melb_weather.csv")

