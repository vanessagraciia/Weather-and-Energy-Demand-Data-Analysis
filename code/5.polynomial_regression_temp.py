import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train_x = pd.read_csv('melb_weather_normalized.csv')
data_train_y = pd.read_csv('pdd_melb_normalized.csv')
x_cols = "Mean Temperature (°C)"
#x_cols = ["Mean Temperature (°C)", "Evaporation (mm)", "Speed of maximum wind gust (km/h)"]
y_cols = "Normalized"
X = data_train_x[x_cols]
Y = data_train_y[y_cols]
plt.scatter(X, Y)
plt.savefig("Mean_Temp_Uno.png")
plt.close()

polynomial_features= PolynomialFeatures(degree=2)
X = np.array(X)
X = np.reshape(X, (-1,1))
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)
plt.scatter(X, Y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')
plt.savefig("5.poly_result_temp.png")
plt.close()

# TESTING

data_test_x = pd.read_csv('melb_weather_normalized_testing.csv')
data_test_y = pd.read_csv('pdd_melb_normalized_testing.csv')
x_cols = "Mean Temperature (°C)"
y_cols = "Normalized"
X_test = data_test_x[x_cols]
Y_test = data_test_y[y_cols]


y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_pred)

print('R2', r2)
print('MSE', mse)
plt.scatter(X, Y)
plt.savefig("Mean_Temp_Uno_Test.png")
plt.close()

polynomial_features= PolynomialFeatures(degree=2)
X = np.array(X)
X = np.reshape(X, (-1,1))
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)
plt.scatter(X, Y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')
plt.savefig("5.poly_result_temp_test.png")
plt.close()
