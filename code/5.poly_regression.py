import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train_x = pd.read_csv('melb_weather_normalized.csv')
data_train_y = pd.read_csv('pdd_melb_normalized.csv')
#x_cols = "Minimum temperature (°C)"
x_cols = "Mean Temperature (°C)"
y_cols = "Normalized"
X = data_train_x[x_cols]
Y = data_train_y[y_cols]

data_test_x = pd.read_csv('melb_weather_normalized_testing.csv')
data_test_y = pd.read_csv('pdd_melb_normalized_testing.csv')
X_test = data_test_x[x_cols]
Y_test = data_test_y[y_cols]

X = np.array(X)
X = np.reshape(X, (-1,1))
X_test = np.array(X_test)
X_test = np.reshape(X_test, (-1,1))

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)
polynomial_features= PolynomialFeatures(degree=2)
X_test = polynomial_features.fit_transform(X_test)

model = LinearRegression()
model.fit(x_poly, Y)

# PLOT 
y_poly_pred = model.predict(x_poly)
plt.scatter(X, Y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')
plt.xlabel("Minimum Temperature (Normalized)", fontsize=8)
plt.ylabel("Energy Demand (Normalized)", fontsize=8)
plt.title("Polynomial Regression Model of Mean Temperature against Energy Demand", fontsize=10)
plt.savefig("5.poly_result_temp.png")
plt.close()

# TESTING
y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

print('R2', r2)
print('RMSE', rmse)


#PLOT

plt.scatter(Y_test, y_pred, alpha=0.3)
plt.title('Linear Regression (Predict Total)')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.savefig("Poly_Correct.png")
plt.close()

# RESIDUALS
residuals = Y_test - y_pred
# plot residuals
plt.scatter(y_pred, residuals, alpha=0.3)
# plot the 0 line (we want our residuals close to 0)
plt.plot([min(y_pred), max(y_pred)], [0,0], color='red')
plt.title('Residual Plot')
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.savefig("Residual_png")


