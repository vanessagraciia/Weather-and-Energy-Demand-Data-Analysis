import pandas as pd
import matplotlib.pyplot as plt

# VISUALIZE WEATHER COMPONENTS
melb_df = pd.read_csv("melb_weather.csv")
melb_df.set_index("Date", inplace=True)

# Plot Min Temp
fig, ax= plt.subplots()
plt.plot(melb_df["Minimum temperature (°C)"])
ticks = [x for x in list(melb_df.index.values) if x[-2:] == "01"]
plt.xlabel("Date")
plt.ylabel("Minimum temperature (°C)")
plt.title("Minimum Temperature")
ax.set_xticks(ticks)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.savefig("melb_min_temp.png")
plt.close()

# Plot Mean Temp
fig, ax= plt.subplots()
plt.plot(melb_df["Mean Temperature (°C)"])
# ticks = [x for x in list(melb_df.index.values) if x[-2:] == "01"]
plt.xlabel("Date")
plt.ylabel("Mean temperature (°C)")
plt.title("Mean Temperature")
# ax.set_xticks(ticks)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.savefig("melb_mean_temp.png")
plt.close()


# VISUALIZE DAILY PRICE DEMAND FLUCTUATIONS
pdd_df = pd.read_csv("price_demand_data.csv")
pdd_melb = pdd_df[pdd_df["REGION"] == "VIC1"]

one_day = pdd_melb[0:48]
fig, ax= plt.subplots()
plt.plot(one_day["SETTLEMENTDATE"], one_day["TOTALDEMAND"])
ticks = [x for x in list(one_day["SETTLEMENTDATE"]) if x[8:10] == "01"]
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.title("Daily Price Demand Fluctuations")
ax.set_xticks(ticks)
plt.savefig("daily_fluctuations.png") 
plt.close()

another_day = pdd_melb[2448:2496]
fig, ax= plt.subplots()
plt.plot(another_day["SETTLEMENTDATE"], another_day["TOTALDEMAND"])
plt.savefig("daily_fluctuations2.png")
plt.close()

            ## NORMALIZED PLOT ##

# VISUALIZE MEAN TEMPERATURE AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
temp_demand = pd.merge(melb_df[["Date", "Mean Temperature (°C)"]], pdd_melb_df, how='inner')
temp_demand.set_index("Date", inplace=True)
#temp_demand.to_csv("norm_temp_demand_melb.csv")
 
plt.scatter(temp_demand["Mean Temperature (°C)"], temp_demand["Normalized"])
plt.xlabel("Mean temperature (°C)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_temp_demand_melb.png")
plt.close()

# VISUALIZE MIN TEMPERATURE AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
mintemp_demand = pd.merge(melb_df[["Date", "Minimum temperature (°C)"]], pdd_melb_df, how='inner')
mintemp_demand.set_index("Date", inplace=True)
#mintemp_demand.to_csv("norm_mintemp_demand_melb.csv")
 
plt.scatter(mintemp_demand["Minimum temperature (°C)"], mintemp_demand["Normalized"])
plt.xlabel("Min temperature (°C)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_mintemp_demand_melb.png")
plt.close()

# VISUALIZE WINDSPEED AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
windspeed_demand = pd.merge(melb_df[["Date", "Speed of maximum wind gust (km/h)"]], pdd_melb_df, how='inner')
windspeed_demand.set_index("Date", inplace=True)
#windspeed_demand.to_csv("norm_windspeed_demand_melb.csv")
 
plt.scatter(windspeed_demand["Speed of maximum wind gust (km/h)"], windspeed_demand["Normalized"])
plt.xlabel("Speed of Maximum Wind Gust (km/h)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_windspeed_demand_melb.png")
plt.close()

# VISUALIZE SUNSHINE AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
sunshine_demand = pd.merge(melb_df[["Date", "Sunshine (hours)"]], pdd_melb_df, how='inner')
sunshine_demand.set_index("Date", inplace=True)
#sunshine_demand.to_csv("norm_sunshine_demand_melb.csv")
 
plt.scatter(sunshine_demand["Sunshine (hours)"], sunshine_demand["Normalized"])
plt.xlabel("Sunshine (hours)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_sunshine_demand_melb.png")
plt.close()

# VISUALIZE RAINFALL AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
rainfall_demand = pd.merge(melb_df[["Date", "Rainfall (mm)"]], pdd_melb_df, how='inner')
rainfall_demand.set_index("Date", inplace=True)
#rainfall_demand.to_csv("norm_rainfall_demand_melb.csv")
 
plt.scatter(rainfall_demand["Rainfall (mm)"], rainfall_demand["Normalized"])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_rainfall_demand_melb.png")
plt.close()

# VISUALIZE EVAPORATION AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
evaporation_demand = pd.merge(melb_df[["Date", "Evaporation (mm)"]], pdd_melb_df, how='inner')
evaporation_demand.set_index("Date", inplace=True)
#evaporation_demand.to_csv("norm_evaporation_demand_melb.csv")
 
plt.scatter(evaporation_demand["Evaporation (mm)"], evaporation_demand["Normalized"])
plt.xlabel("Evaporation (mm)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("norm_evaporation_demand_melb.png")
plt.close()

            ## UNNORMALIZED PLOT ##
            
# VISUALIZE MEAN TEMPERATURE AGAINST PRICE DEMAND 
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
temp_demand = pd.merge(melb_df[["Date", "Mean Temperature (°C)"]], pdd_melb_df, how='inner')
temp_demand.set_index("Date", inplace=True)
#temp_demand.to_csv("unnorm_temp_demand_melb.csv")
 
plt.scatter(temp_demand["Mean Temperature (°C)"], temp_demand["TOTAL DEMAND"])
plt.xlabel("Mean temperature (°C)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_temp_demand_melb.png")
plt.close()

# VISUALIZE MIN TEMPERATURE AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
mintemp_demand = pd.merge(melb_df[["Date", "Minimum temperature (°C)"]], pdd_melb_df, how='inner')
mintemp_demand.set_index("Date", inplace=True)
#mintemp_demand.to_csv("unnorm_mintemp_demand_melb.csv")
 
plt.scatter(mintemp_demand["Minimum temperature (°C)"], mintemp_demand["TOTAL DEMAND"])
plt.xlabel("Min temperature (°C)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_mintemp_demand_melb.png")
plt.close()

# VISUALIZE WINDSPEED AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
windspeed_demand = pd.merge(melb_df[["Date", "Speed of maximum wind gust (km/h)"]], pdd_melb_df, how='inner')
windspeed_demand.set_index("Date", inplace=True)
#windspeed_demand.to_csv("unnorm_windspeed_demand_melb.csv")
 
plt.scatter(windspeed_demand["Speed of maximum wind gust (km/h)"], windspeed_demand["TOTAL DEMAND"])
plt.xlabel("Speed of Maximum Wind Gust (km/h)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_windspeed_demand_melb.png")
plt.close()

# VISUALIZE SUNSHINE AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
sunshine_demand = pd.merge(melb_df[["Date", "Sunshine (hours)"]], pdd_melb_df, how='inner')
sunshine_demand.set_index("Date", inplace=True)
#sunshine_demand.to_csv("unnorm_sunshine_demand_melb.csv")
 
plt.scatter(sunshine_demand["Sunshine (hours)"], sunshine_demand["TOTAL DEMAND"])
plt.xlabel("Sunshine (hours)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_sunshine_demand_melb.png")
plt.close()

# VISUALIZE RAINFALL AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
rainfall_demand = pd.merge(melb_df[["Date", "Rainfall (mm)"]], pdd_melb_df, how='inner')
rainfall_demand.set_index("Date", inplace=True)
#rainfall_demand.to_csv("unnorm_rainfall_demand_melb.csv")
 
plt.scatter(rainfall_demand["Rainfall (mm)"], rainfall_demand["TOTAL DEMAND"])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_rainfall_demand_melb.png")
plt.close()

# VISUALIZE EVAPORATION AGAINST PRICE DEMAND
melb_df = pd.read_csv("melb_weather.csv")
pdd_melb_df = pd.read_csv("pdd_melb.csv")
evaporation_demand = pd.merge(melb_df[["Date", "Evaporation (mm)"]], pdd_melb_df, how='inner')
evaporation_demand.set_index("Date", inplace=True)
#evaporation_demand.to_csv("unnorm_evaporation_demand_melb.csv")
 
plt.scatter(evaporation_demand["Evaporation (mm)"], evaporation_demand["TOTAL DEMAND"])
plt.xlabel("Evaporation (mm)")
plt.ylabel("Total Energy Demand")
plt.title("Victoria")
plt.subplots_adjust(bottom=0.3)
plt.savefig("unnorm_evaporation_demand_melb.png")
plt.close()
