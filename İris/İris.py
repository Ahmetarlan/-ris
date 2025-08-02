import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("C:\\Users\\knox0\\Desktop\\archive\\Iris.csv")
df = pd.DataFrame(df)
print(df.columns.tolist())

print(df.head())
print(df.info())
print(df.describe())
print(df.drop(columns="Id", inplace=True))
print(df.head())
sns.pairplot(df)
plt.show()


print(df["Species"].value_counts())

label = LabelEncoder()
df["Species"] = label.fit_transform(df["Species"])
print(df.head())
print(df.tail())


X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

svc = SVC(kernel="rbf")
svc.fit(X_train_sc,y_train)
y_pred = svc.predict(X_test_sc)

matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
classi = classification_report(y_test, y_pred)
print("Confusion matrix:\n", matrix)
print("Accuracy:", acc)
print("Classification report:\n", classi)


############


logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train_sc,y_train)
y_pred_logi = logistic.predict(X_test_sc)



matrix = confusion_matrix(y_test, y_pred_logi)
acc = accuracy_score(y_test, y_pred_logi)
classi = classification_report(y_test, y_pred_logi)
print("Confusion matrix logi:\n", matrix)
print("Accuracy logi:", acc)
print("Classification report logi:\n", classi)

############

params = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [100, 10, 1, 0.1, 0.01],  # Doğru parametre adı büyük C olmalı
    "solver": ["lbfgs","liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
}

strati = StratifiedKFold()

grid = GridSearchCV(estimator=logistic,param_grid=params,cv=strati,scoring="accuracy", n_jobs=-1)
grid.fit(X_train_sc,y_train)

print(grid.best_estimator_)
print(grid.best_score_)

y_pred_grid =grid.predict(X_test_sc)


matrix = confusion_matrix(y_test, y_pred_grid)
acc = accuracy_score(y_test, y_pred_grid)
classi = classification_report(y_test, y_pred_grid)
print("Confusion matrix grid:\n", matrix)
print("Accuracy grid:", acc)
print("Classification report grid:\n", classi)

########