import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from statistics import mean

# #Get General Statistics (From Non-Normalized Dataset)
# melb_nn_df = pd.read_csv("melb_weather.csv")
# melb_pdd_nn_df = pd.read_csv("pdd_melb.csv")
# melb_nn = pd.merge(melb_nn_df, melb_pdd_nn_df, how='inner')
# stats = melb_nn.describe()
# melb_nn.drop("Unnamed: 0", inplace=True, axis=1)

# melb_df = pd.read_csv("melb_weather_normalized.csv")
# melb_pdd_df = pd.read_csv("pdd_melb_normalized.csv")
# melb = pd.merge(melb_df, melb_pdd_df, how='inner')
# melb.set_index("Date", drop=True, inplace=True)

# # NOTE: NOT LINEAR
# # Find Correlation Between Each Weather Component and Total Demand
# corr_data = melb.corr()["TOTAL DEMAND"]

# # FEATURE SELECTION BASED ON MUTUAL INFORMATION

# # Discretize continuous data

# num = 30
# features_dict = {"Minimum temperature (°C)": num, "Maximum temperature (°C)": num, "Mean Temperature (°C)": num, "3pm Temperature (°C)": num, "9am Temperature (°C)": num, "Rainfall (mm)": 10, "Evaporation (mm)": 15, "Sunshine (hours)": 8, "Speed of maximum wind gust (km/h)": 10, "TOTAL DEMAND": 45}
# for feature in melb_nn.iloc[:,1:]:
#     equal_width = KBinsDiscretizer(n_bins=features_dict[feature], encode='ordinal', strategy='uniform')
#     melb["Binned " + feature] = equal_width.fit_transform(melb[feature].to_numpy().reshape(-1, 1)).astype(int)

# #Calculate mutual information
# filtered_features = []
# THRESHOLD = 0.2 
# features = melb[["Binned Minimum temperature (°C)", "Binned Maximum temperature (°C)", "Binned Mean Temperature (°C)", "Binned Rainfall (mm)", "Binned Evaporation (mm)", "Binned Sunshine (hours)", "Binned Speed of maximum wind gust (km/h)", "Binned TOTAL DEMAND"]]
# class_label = melb["Binned TOTAL DEMAND"]
# mi_arr = mutual_info_classif(X=features, y=class_label, discrete_features=True)
# for feature, mi in zip(features.columns, mi_arr):
#     # print(f'MI value for feature "{feature}": {mi:.4f}')
#     if(mi >= THRESHOLD): 
#         filtered_features.append(feature)

# # print('\nFeature set after filtering with MI:', filtered_features)

# filtered_data = []
# for feature in filtered_features:
#     filtered_data.append(melb[feature])
# filtered_melb = (pd.DataFrame(filtered_data)).T
# demand = melb["Binned TOTAL DEMAND"]
# filtered_melb.insert(0, "Binned_Total_Demand", demand)
# filtered_melb.to_csv("manual_bins.csv")

# # FIND ACCURACY

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.model_selection import KFold 
# import numpy as np 
# import matplotlib.pyplot as plt 
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from statistics import mean
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# kf_CV = KFold(n_splits=10, shuffle=True, random_state=42)
# data_train = pd.read_csv('manual_bins.csv')
# x_cols = ["Binned Mean Temperature (°C)", "Binned Rainfall (mm)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)"]
# y_cols = "Binned TOTAL DEMAND"
# X = data_train[x_cols]
# Y = data_train[y_cols]

# # KNN with n = 1
# results = []
# for train_index, test_index in kf_CV.split(X):
#     X_train = []
#     X_test = []
#     Y_train = []
#     Y_test = []
#     for index in train_index:
#         X_train.append(X.iloc[index])
#         X_test.append(X.iloc[index])
#         Y_train.append(Y.iloc[index])
#         Y_test.append(Y.iloc[index])
            
#     # Training
#     knn = KNN(n_neighbors=1)
#     knn.fit(X_train, Y_train)    
    
#     # Predictions
#     y_pred = knn.predict(X_test)
#     results.append(accuracy_score(Y_test, y_pred))
# print("n = 1: ", mean(results))

# # KNN with n = 2
# results = []
# for train_index, test_index in kf_CV.split(X):
#     X_train = []
#     X_test = []
#     Y_train = []
#     Y_test = []
#     for index in train_index:
#         X_train.append(X.iloc[index])
#         X_test.append(X.iloc[index])
#         Y_train.append(Y.iloc[index])
#         Y_test.append(Y.iloc[index])

#     # Training
#     knn = KNN(n_neighbors=2)
#     knn.fit(X_train, Y_train)    
    
#     # Predictions
#     y_pred = knn.predict(X_test)
#     results.append(accuracy_score(Y_test, y_pred))
# print("n = 2: ", mean(results))

# # KNN with n = 3
# results = []
# for train_index, test_index in kf_CV.split(X):
#     X_train = []
#     X_test = []
#     Y_train = []
#     Y_test = []
#     for index in train_index:
#         X_train.append(X.iloc[index])
#         X_test.append(X.iloc[index])
#         Y_train.append(Y.iloc[index])
#         Y_test.append(Y.iloc[index])

#     # Training
#     knn = KNN(n_neighbors=3)
#     knn.fit(X_train, Y_train)    
    
#     # Predictions
#     y_pred = knn.predict(X_test)
#     results.append(accuracy_score(Y_test, y_pred))
# print("n = 3: ", mean(results))

# # KNN with n = 5
# results = []
# for train_index, test_index in kf_CV.split(X):
#     X_train = []
#     X_test = []
#     Y_train = []
#     Y_test = []
#     for index in train_index:
#         X_train.append(X.iloc[index])
#         X_test.append(X.iloc[index])
#         Y_train.append(Y.iloc[index])
#         Y_test.append(Y.iloc[index])

#     # Training
#     knn = KNN(n_neighbors=5)
#     knn.fit(X_train, Y_train)    
    
#     # Predictions
#     y_pred = knn.predict(X_test)
#     results.append(accuracy_score(Y_test, y_pred))
# print("n = 5: ", mean(results))
    
# # # TESTING
# # # Pick KNN n = 1

# test = pd.read_csv('NEW_BINNED_AND_FILTERED_DATA_TESTING.csv')
# x_cols = x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)"]
# y_cols = "Binned TOTAL DEMAND"

# x_test = test[x_cols]
# y_test = test[y_cols]

# knn = KNN(n_neighbors=3)
# knn.fit(X, Y)
# y_pred = knn.predict(x_test)
# accuracy = knn.score(x_test, y_test)
# print('Accuracy: ', accuracy)
#print('Recall: ', recall_score(y_test, y_pred, average='macro'))
# print('Precision: ', precision_score(y_test, y_pred, average="macro"))
# print('F1: ', f1_score(y_test, y_pred, average="macro"))

# cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_).plot()
# plt.title('Confusion Matrix')
# plt.savefig("confusion_matrix.png")

melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
melb_n = pd.merge(melb_df, pdd_melb_df, how='inner')
# Select only filtered features
melb_n =  melb_n[['Minimum temperature (°C)', 'Mean Temperature (°C)', 'Maximum temperature (°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Speed of maximum wind gust (km/h)', 'TOTAL DEMAND']]
features_dict = {'Minimum temperature (°C)': 5, "Mean Temperature (°C)": 5, "Maximum temperature (°C)": 5, "Rainfall (mm)": 5, "Evaporation (mm)": 5, "Speed of maximum wind gust (km/h)": 5, "TOTAL DEMAND": 5}
