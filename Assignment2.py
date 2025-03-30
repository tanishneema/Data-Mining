import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

TrainD = pd.read_csv("train.csv")
TestD = pd.read_csv("test.csv")

Data = [TrainD, TestD]

for dset in Data:
    dset['Title'] = dset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
    dset['Title'] = dset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
    dset['Title'] = dset['Title'].replace('Mlle', 'Miss')
    dset['Title'] = dset['Title'].replace('Ms', 'Miss')
    dset['Title'] = dset['Title'].replace('Mme', 'Mrs')

    dset['Sex'] = dset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    dset['Embarked'] = dset['Embarked'].fillna(dset['Embarked'].mode()[0])
    dset['Embarked'] = dset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    dset['Fare'] = dset['Fare'].fillna(dset['Fare'].median())
    dset['FareBand'] = pd.qcut(dset['Fare'], 4, labels=False)

    dset['Age'] = dset['Age'].fillna(dset['Age'].median())
    dset['AgeBand'] = pd.cut(dset['Age'], 5, labels=False)

    dset['FamilySize'] = dset['SibSp'] + dset['Parch'] + 1
    dset['IsAlone'] = 0
    dset.loc[dset['FamilySize'] == 1, 'IsAlone'] = 1

    dset.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare', 'Age'], axis=1, inplace=True, errors='ignore')

    le = LabelEncoder()
    dset['Title'] = le.fit_transform(dset['Title'])

XTrain = TrainD.drop("Survived", axis=1)
yTrain = TrainD["Survived"]

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(XTrain, yTrain)

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=XTrain.columns, class_names=["Not Survived", "Survived"], filled=True)
plt.show()

score = cross_val_score(clf, XTrain, yTrain, cv=5)
print("Decision Tree Average Accuracy:", score.mean())

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
score_rf = cross_val_score(rf, XTrain, yTrain, cv=5)
print("Random Forest Average Accuracy:", score_rf.mean())

svmD = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
L = np.array([-1, 1, 1, -1])

mappedSVM = np.array([[x[0], x[0]*x[1]] for x in svmD])

plt.figure()
for i, point in enumerate(mappedSVM):
    if L[i] == 1:
        plt.scatter(point[0], point[1], color='blue', marker='o')
    else:
        plt.scatter(point[0], point[1], color='red', marker='x')

plt.plot([-2, 2], [1, 1], 'k--')
plt.plot([-2, 2], [-1, -1], 'k--')
plt.plot([-2, 2], [0, 0], 'k')

plt.xlabel("x1")
plt.ylabel("x1 * x2")
plt.title("Mapped Points and Max-Margin Separator")
plt.grid(True)
plt.show()

print("Margin:", 1.0)
