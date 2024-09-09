# # automatic nested cross-validation for random forest on a classification dataset
# from numpy import mean
# import pandas as pd
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# kf_CV = KFold(n_splits=10, shuffle=True, random_state=42)
# data_train = pd.read_csv('filtered_melb_weather.csv')
# #x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)", "Binned Rainfall (mm)"]
# x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)"]
# #x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Rainfall (mm)"]
# y_cols = "Binned TOTAL DEMAND"
# X = data_train[x_cols]
# Y = data_train[y_cols]

# # create dataset

# # define the model
# model = RandomForestClassifier(random_state=42)
# # define search space
# space = dict()
# space[x_cols] = X
# space[y_cols] = Y
# #print(space)

# # define search

# search = GridSearchCV(model, scoring='accuracy', n_jobs=1, cv=kf_CV, refit=True)
# # configure the cross-validation procedure
# cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
# # execute the nested cross-validation
# scores = cross_val_score(search, X, Y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score

imbalance = pd.read_csv('filtered_melb_weather.csv')
# Check the class label to see that it is imbalanced classification (N=400)
print(imbalance['Binned TOTAL DEMAND'].value_counts())
imbalance.hist('Binned TOTAL DEMAND')
# plt.show()

# Get X, y
X_imb = imbalance['Binned Mean Temperature (°C)']
y_imb = np.array(imbalance['Binned TOTAL DEMAND'])
X_imb2 = pd.DataFrame(X_imb).to_numpy()

accuracies = []
precisions = []
recalls = []
f1s = []

n = X_imb2.shape[0]
dataidx = range(n)
k = 10

bs = []
for k in range(10):
    # prepare bootstrap sample
    boot_index = resample(range(n), replace=True, n_samples=n, random_state=k)
    # out of bag observations
    oob_index = [x for x in range(n) if x not in boot_index]
    # Split datasets
    X_imb_train = X_imb2[boot_index,:]
    X_imb_test = X_imb2[oob_index,:]
    y_imb_train = y_imb[boot_index]
    y_imb_test = y_imb[oob_index]
    
    # Train
    knn = KNN(n_neighbors=10)
    knn.fit(X_imb_train, y_imb_train)
    
    # Predict
    y_imb_pred=knn.predict(X_imb_test)

    # Evaluate
    accuracies.append(accuracy_score(y_imb_test, y_imb_pred))
    recalls.append(recall_score(y_imb_test, y_imb_pred, average='micro', labels=np.unique(y_imb_pred)))
    precisions.append(precision_score(y_imb_test, y_imb_pred, average='micro', labels=np.unique(y_imb_pred)))
    f1s.append(f1_score(y_imb_test, y_imb_pred, average='micro', labels=np.unique(y_imb_pred)))

print("Accuracy from each bootstrap sample:", accuracies)
#Display average of accuracy scores
avg_acc_score = np.mean(accuracies)
print("Mean accuracy from all bootstrap samples:", avg_acc_score)

print("Precision from each bootstrap sample:", precisions)
#Display average of precision scores
avg_precision_score = np.mean(precisions)
print("Mean precision from all bootstrap samples:", avg_precision_score)

print("Recall from each bootstrap sample:", recalls)
#Display average of recall scores
avg_recall_score = np.mean(recalls)
print("Mean recall from all bootstrap samples:", avg_recall_score)
