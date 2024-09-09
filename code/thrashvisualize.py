import pandas as pd
import matplotlib.pyplot as plt
pdd_df = pd.read_csv("price_demand_data.csv")

#NSW plot
nsw_pdd_df = pdd_df.loc[pdd_df["REGION"]=="NSW1"]
nsw_pdd_df = (nsw_pdd_df[nsw_pdd_df.columns[2]]).reset_index(drop=True)
nsw_pdd_df.plot.line()
plt.savefig("nsw_pdd.png")
plt.clf()

#VIC plot
vic_pdd_df = pdd_df.loc[pdd_df["REGION"]=="VIC1"]
vic_pdd_df = (vic_pdd_df[vic_pdd_df.columns[2]]).reset_index(drop=True)
vic_pdd_df.plot.line(x="Time", y="Price Demand")
plt.savefig("vic_pdd.png")
plt.clf()

#SA plot
sa_pdd_df = pdd_df.loc[pdd_df["REGION"]=="SA1"]
sa_pdd_df = (sa_pdd_df[sa_pdd_df.columns[2]]).reset_index(drop=True)
sa_pdd_df.plot.line(x="Time", y="Price Demand")
plt.savefig("sa_pdd.png")
plt.clf()

#QLD plot
qld_pdd_df = pdd_df.loc[pdd_df["REGION"]=="QLD1"]
qld_pdd_df = (qld_pdd_df[qld_pdd_df.columns[2]]).reset_index(drop=True)
qld_pdd_df.plot.line(x="Time", y="Price Demand")
plt.savefig("qld_pdd.png")
plt.clf()

#Sydney Temperature
syd_df = pd.read_csv("weather_sydney.csv")
syd_df[syd_df.columns[1]].plot.line()
syd_df[syd_df.columns[2]].plot.line()
plt.savefig("syd_min_max_temp.png")
plt.clf()

range_syd_df = syd_df[syd_df.columns[2]] - syd_df[syd_df.columns[1]]
range_syd_df.plot.line()
plt.savefig("syd_range_temp.png")
plt.clf()

#Sydney Rainfall
syd_df[syd_df.columns[3]].plot.line()
plt.savefig("syd_rainfall.png")
plt.clf()




