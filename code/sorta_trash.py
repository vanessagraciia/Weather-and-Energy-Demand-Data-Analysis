import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
import numpy as np 
from statistics import mean
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train_x = pd.read_csv('melb_weather_normalized.csv')
data_train_y = pd.read_csv('pdd_melb_normalized.csv')
x_cols = "Mean Temperature (°C)"
y_cols = "Normalized"
X = data_train_x[x_cols]
Y = data_train_y[y_cols]

kf_CV = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
r2_list= []
for train_index, test_index in kf_CV.split(X):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for index in train_index:
        X_train.append(X.iloc[index])
        X_test.append(X.iloc[index])
        Y_train.append(Y.iloc[index])
        Y_test.append(Y.iloc[index])

    polynomial_features= PolynomialFeatures(degree=2)
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (-1,1))
    x_poly = polynomial_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(x_poly, Y_train)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(Y_train,y_poly_pred))
    r2 = r2_score(Y_train,y_poly_pred)
    rmse_list.append(rmse)
    r2_list.append(r2)

    plt.scatter(X_train, Y_train, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
    X, y_poly_pred = zip(*sorted_zip)
    plt.plot(X, y_poly_pred, color='m')
    plt.savefig("5.poly_result.png")
    print("rmse: ", mean(rmse_list))

print("r2: ", mean(r2_list))
