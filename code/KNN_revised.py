import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statistics import mean
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

kf_CV = KFold(n_splits=10, shuffle=True, random_state=42)
data_train = pd.read_csv('NEW_BINNED_AND_FILTERED_DATA.csv')
x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)"]
y_cols = "Binned TOTAL DEMAND"
X = data_train[x_cols]
Y = data_train[y_cols]

# KNN with n = 1
results = []
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
            
    # Training
    knn = KNN(n_neighbors=1)
    knn.fit(X_train, Y_train)    
    
    # Predictions
    y_pred = knn.predict(X_test)
    results.append(accuracy_score(Y_test, y_pred))
print("n = 1: ", mean(results))

# KNN with n = 2
results = []
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

    # Training
    knn = KNN(n_neighbors=2)
    knn.fit(X_train, Y_train)    
    
    # Predictions
    y_pred = knn.predict(X_test)
    results.append(accuracy_score(Y_test, y_pred))
print("n = 2: ", mean(results))

# KNN with n = 3
results = []
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

    # Training
    knn = KNN(n_neighbors=3)
    knn.fit(X_train, Y_train)    
    
    # Predictions
    y_pred = knn.predict(X_test)
    results.append(accuracy_score(Y_test, y_pred))
print("n = 3: ", mean(results))


# KNN with n = 5
results = []
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

    # Training
    knn = KNN(n_neighbors=5)
    knn.fit(X_train, Y_train)    
    
    # Predictions
    y_pred = knn.predict(X_test)
    results.append(accuracy_score(Y_test, y_pred))
print("n = 5: ", mean(results))
    
# # TESTING
# # Pick KNN n = 1

test = pd.read_csv('NEW_BINNED_AND_FILTERED_DATA_TESTING.csv')
x_cols = x_cols = ["Binned Mean Temperature (°C)", "Binned Evaporation (mm)", "Binned Speed of maximum wind gust (km/h)"]
y_cols = "Binned TOTAL DEMAND"

x_test = test[x_cols]
y_test = test[y_cols]

knn = KNN(n_neighbors=5)
knn.fit(X, Y)
y_pred = knn.predict(x_test)
accuracy = knn.score(x_test, y_test)
print('Accuracy: ', accuracy)
#print('Recall: ', recall_score(y_test, y_pred, average='macro'))
# print('Precision: ', precision_score(y_test, y_pred, average="macro"))
# print('F1: ', f1_score(y_test, y_pred, average="macro"))

# cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_).plot()
# plt.title('Confusion Matrix')
# plt.savefig("confusion_matrix.png")
