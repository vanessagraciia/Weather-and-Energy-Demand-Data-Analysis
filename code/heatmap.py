import numpy as np 
from pandas import DataFrame
import seaborn as sns
import csv
import pandas as pd

melb_df = pd.read_csv('melb_weather_normalized.csv')
pdd_melb_df = pd.read_csv('pdd_melb_normalized.csv')
temp_demand = pd.merge(melb_df[["Date", "Mean Temperature (°C)"]], pdd_melb_df, how='inner')
temp_demand.set_index("Date", inplace=True)
dictionaryInstance = temp_demand.to_dict(orient="list")

Index = dictionaryInstance["Mean Temperature (°C)"]  
Cols = dictionaryInstance["TOTAL DEMAND"]
df = DataFrame(abs(np.random.randn(len(Index), len(Cols))), index=Index, columns=Cols)

sns.heatmap(df, annot=True)
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()

