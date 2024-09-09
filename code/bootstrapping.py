from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score

melb_df = pd.read_csv("melb_weather_normalized.csv")
pdd_melb_df = pd.read_csv("pdd_melb_normalized.csv")
melb_n = pd.merge(melb_df, pdd_melb_df, how='inner')
# Select only filtered features
melb_n =  melb_n[['Minimum temperature (°C)', 'Mean Temperature (°C)', 'Maximum temperature (°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Speed of maximum wind gust (km/h)', 'TOTAL DEMAND']]
features_dict = {'Minimum temperature (°C)': 5, "Mean Temperature (°C)": 5, "Maximum temperature (°C)": 5, "Rainfall (mm)": 5, "Evaporation (mm)": 5, "Speed of maximum wind gust (km/h)": 5, "TOTAL DEMAND": 5}
#features_dict = {'Minimum temperature (°C)': 10, "Mean Temperature (°C)": 10, "Maximum temperature (°C)": 10, "Rainfall (mm)": 10, "Evaporation (mm)": 10, "Speed of maximum wind gust (km/h)": 10, "TOTAL DEMAND": 10}

for feature in melb_n.iloc[:,0:]:
    equal_width = KBinsDiscretizer(n_bins=features_dict[feature], encode='ordinal', strategy='uniform')
    melb_n["Binned " + feature] = equal_width.fit_transform(melb_n[feature].to_numpy().reshape(-1, 1)).astype(int)

# Get X, y
X_imb = melb_n['Binned Mean Temperature (°C)']
y_imb = np.array(melb_n['Binned TOTAL DEMAND'])
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
