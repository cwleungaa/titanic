import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

import seaborn as sns

import collections, operator

training_data_path = "./data_set/train.csv"

test_data_path = "./data_set/test.csv"

#Creating training df and testing df
training_data_df = pd.read_csv(training_data_path)

test_data_df = pd.read_csv(test_data_path)

print("The summary of null data in training data:")
print(training_data_df.isnull().sum())  
# Cabin: 687, Age: 177, Embarked: 2

print("The summary of null data in test data:")
print(test_data_df.isnull().sum())   
# Cabin: 327, Age: 86, Fare: 1


# Preprocessing

# Filling 'Age' and 'Fare' NaN value by median

training_data_df.Age.fillna(training_data_df.Age.median(), inplace=True)
test_data_df.Age.fillna(training_data_df.Age.median(), inplace=True)
test_data_df.Fare.fillna(training_data_df.Fare.median(), inplace=True)

# Change the feature 'Cabin' to 'HasCabin'

training_data_df.Cabin = training_data_df.Cabin.notnull().astype("int")
test_data_df.Cabin = test_data_df.Cabin.notnull().astype("int")

# Add a feature for whether passager is 'FamilySize'

training_data_df["FamilySize"] = training_data_df.loc[:, ["SibSp", "Parch"]].sum(axis=1)
test_data_df["FamilySize"] = test_data_df.loc[:, ["SibSp", "Parch"]].sum(axis=1)

# Add a feature for whether passager is 'Alone'

training_data_df["Alone"] = training_data_df.FamilySize.apply(lambda x : 1 if x != 0 else 0)
test_data_df["Alone"] = test_data_df.FamilySize.apply(lambda x : 1 if x != 0 else 0)

# Handle Embarked with simpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

training_data_df["Embarked"] = imputer.fit_transform(np.array(training_data_df["Embarked"]).reshape([-1,1]))
test_data_df["Embarked"] = imputer.fit_transform(np.array(test_data_df["Embarked"]).reshape([-1,1]))

training_data_df["Embarked"] = training_data_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})
test_data_df["Embarked"] = test_data_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2})

# Not contain NaN data anymore
print("Nan value checking:")
print(training_data_df.isnull().sum())

# Map Sex to 0,1

training_sex_pd = pd.get_dummies(training_data_df.Sex, drop_first=True)
test_sex_pd = pd.get_dummies(test_data_df.Sex, drop_first=True)

training_data_df["Male"] = training_sex_pd.male
test_data_df["Male"] = test_sex_pd.male

# Map Age into group

bins = [0, 2, 18, 35, 65, np.inf]
names = ['<2', '2-18', '18-35', '35-65', '65+']

training_data_df['AgeRange'] = pd.cut(training_data_df['Age'], bins, labels=names)
test_data_df['AgeRange'] = pd.cut(test_data_df['Age'], bins, labels=names)

ord = OrdinalEncoder(categories=[['<2', '2-18', '18-35', '35-65', '65+']], dtype=int)

training_data_df["Age"] = ord.fit_transform(np.array(training_data_df['AgeRange']).reshape(-1,1))
test_data_df["Age"] = ord.fit_transform(np.array(test_data_df['AgeRange']).reshape(-1,1))

# Map Fare into group

fare_bins = [-1, 7.896, 14.454, 31.275, np.inf]
fare_names = ['<8', '8-14', '15-30', '31+']

training_data_df['FareRange'] = pd.cut(training_data_df['Fare'], fare_bins, labels=fare_names)
test_data_df['FareRange'] = pd.cut(test_data_df['Fare'], fare_bins, labels=fare_names)

fare_ord = OrdinalEncoder(categories=[['<8', '8-14', '15-30', '31+']], dtype=int)
training_data_df['Fare'] = fare_ord.fit_transform(np.array(training_data_df['FareRange']).reshape(-1,1))
test_data_df['Fare'] = fare_ord.fit_transform(np.array(test_data_df['FareRange']).reshape(-1,1))

# Feature Scalaring

X = training_data_df[["Pclass", "Age", "Male", "SibSp", "Parch","Fare", "Cabin", "FamilySize", "Alone", "Embarked"]]

X_testing_set = test_data_df[["Pclass", "Age", "Male", "SibSp", "Parch","Fare", "Cabin", "FamilySize", "Alone", "Embarked"]]

y = training_data_df[['Survived']]

# Using MinMaxScaler() to resize the data in range of [0,1]
fitted_obj = MinMaxScaler().fit(X)

transformed_X = fitted_obj.transform(X)

transformed_X_testing = fitted_obj.transform(X_testing_set)

transformed_X_df = pd.DataFrame(transformed_X, columns=X.columns)

transformed_X_testing_df = pd.DataFrame(transformed_X_testing, columns=X_testing_set.columns)

# Plot the heatmap graph to show the correlation:
sns.set(font_scale=1.1)
correlation_train = training_data_df.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.01f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False).set_title("Heatmap for training_data_df")

plt.show()


print(transformed_X_df.head(5))
print("-------------transformed_X_df--------------")
#    Pclass   Age  Male  SibSp  Parch  Fare  Cabin  FamilySize  Alone
# 0     1.0  0.50   1.0  0.125    0.0   0.0    0.0         0.1    1.0
# 1     0.0  0.75   0.0  0.125    0.0   0.6    1.0         0.1    1.0
# 2     1.0  0.50   0.0  0.000    0.0   0.0    0.0         0.0    0.0
# 3     0.0  0.50   0.0  0.125    0.0   0.4    1.0         0.1    1.0
# 4     1.0  0.50   1.0  0.000    0.0   0.0    0.0         0.0    0.0

print("-------------transformed_X_te--------------")

print(transformed_X_testing_df.head(5))

# -------------test X--------------
#    Pclass   Age  Male  SibSp     Parch  Fare  Cabin  FamilySize  Alone
# 0     1.0  0.50   1.0  0.000  0.000000   0.0    0.0         0.0    0.0
# 1     1.0  0.75   0.0  0.125  0.000000   0.0    0.0         0.1    1.0
# 2     0.5  0.75   1.0  0.000  0.000000   0.0    0.0         0.0    0.0
# 3     1.0  0.50   1.0  0.000  0.000000   0.0    0.0         0.0    0.0
# 4     1.0  0.50   0.0  0.125  0.166667   0.0    0.0         0.2    1.0

# Plot a graph to show all data lie in range of (0,1)
f, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data = transformed_X_df)
plt.title("transformed_X_df distribution")
plt.show()

# Split the training dataset 

X_train, X_test, y_train, y_test = train_test_split(transformed_X_df, y, test_size=0.2, random_state=1)

RunResult = collections.namedtuple('RunResult', 'model pred accuracy clf_report conf_mat')

model_list = [
             SVC(kernel='linear'),
             KNeighborsClassifier(n_neighbors=5),
             LogisticRegression(max_iter=5000),
             DecisionTreeClassifier(),
             RandomForestClassifier(),
            ]

result_list = []

for model in model_list:
    model.fit(X_train, y_train.values.ravel())
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    clf_report = metrics.classification_report(y_test, prediction)
    conf_mat = confusion_matrix(y_test, prediction)
    result_list.append(RunResult(model, prediction, accuracy, clf_report, conf_mat))

result_list.sort(key = operator.attrgetter('accuracy'), reverse=True)

# Print the accuracy result list and show in a heatmap
space=' '
sns.set(font_scale=0.9)

fig, axes = plt.subplots(2, 4, figsize=(15,8))
axes = axes.ravel()
for i, result in enumerate(result_list):
    model_name = type(result.model).__name__
    print(f'{i+1:2}. {model_name:30}              {result.accuracy:6.5f}')
    print('-'*55)
    print(f'  {result.clf_report}')
    sns.heatmap(result.conf_mat, square=True, annot=True, fmt='d', cmap='Reds',\
                ax=axes[i]).set_title(model_name)
    axes[i].set_xlabel('Predict')
    axes[i].set_ylabel('Actual')
plt.show()

# Try to tune LogisticRegression Hyperparameters

# model = LogisticRegression(solver='liblinear', max_iter=5000)

# params = [{'C':[0.001, 0.01, 0.1, 1, 10, 100],'penalty': ['l1','l2']},]

# search_cv = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy', return_train_score=True)
# search_cv.fit(X_train, y_train.values.ravel())
# print(f"The search_cv.best_estimator_ is {search_cv.best_estimator_}") 
# #The search_cv.best_estimator_ is LogisticRegression(C=10, max_iter=5000, penalty='l1', solver='liblinear')

# print("GridSearchCV results:\n")
# for i, (mean, params) in enumerate(zip(search_cv.cv_results_['mean_test_score'], search_cv.cv_results_['params'])):
#     if search_cv.best_score_ == mean:
#         print(f' {i+1:2} * {mean:10.5f}, {params}')
#     else:
#         print(f' {i+1:2}   {mean:10.5f}, {params}')  

#  1. LogisticRegression                          0.82123
# -------------------------------------------------------
#                 precision    recall  f1-score   support

#            0       0.84      0.87      0.85       106
#            1       0.80      0.75      0.77        73

#     accuracy                           0.82       179
#    macro avg       0.82      0.81      0.81       179
# weighted avg       0.82      0.82      0.82       179

#  2. RandomForestClassifier                      0.78771
# -------------------------------------------------------
#                 precision    recall  f1-score   support

#            0       0.79      0.87      0.83       106
#            1       0.78      0.67      0.72        73

#     accuracy                           0.79       179
#    macro avg       0.79      0.77      0.77       179
# weighted avg       0.79      0.79      0.78       179

# Try to tune Random forest Hyperparameters

# model = RandomForestClassifier(random_state=1)

# params = {'n_estimators':[64, 128, 256], 'max_depth':[2, 4, 8, 16, 36, 64]}

# search_cv = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy', return_train_score=True)
# search_cv.fit(X_train, y_train.values.ravel())
# print(f"The search_cv.best_estimator_ is {search_cv.best_estimator_}") 
# #The search_cv.best_estimator_ is RandomForestClassifier(max_depth=4, n_estimators=256, random_state=1)

# print("GridSearchCV results:\n")
# for i, (mean, params) in enumerate(zip(search_cv.cv_results_['mean_test_score'], search_cv.cv_results_['params'])):
#     if search_cv.best_score_ == mean:
#         print(f' {i+1:2} * {mean:10.5f}, {params}')
#     else:
#         print(f' {i+1:2}   {mean:10.5f}, {params}')  

#   1      0.80201, {'max_depth': 2, 'n_estimators': 64}
#   2      0.79920, {'max_depth': 2, 'n_estimators': 128}
#   3      0.79500, {'max_depth': 2, 'n_estimators': 256}
#   4      0.81891, {'max_depth': 4, 'n_estimators': 64}
#   5      0.82451, {'max_depth': 4, 'n_estimators': 128}
#   6 *    0.82872, {'max_depth': 4, 'n_estimators': 256}
#   7      0.81041, {'max_depth': 8, 'n_estimators': 64}
#   8      0.80480, {'max_depth': 8, 'n_estimators': 128}
#   9      0.81182, {'max_depth': 8, 'n_estimators': 256}
#  10      0.80337, {'max_depth': 16, 'n_estimators': 64}
#  11      0.80760, {'max_depth': 16, 'n_estimators': 128}
#  12      0.80901, {'max_depth': 16, 'n_estimators': 256}
#  13      0.80197, {'max_depth': 36, 'n_estimators': 64}
#  14      0.80760, {'max_depth': 36, 'n_estimators': 128}
#  15      0.80901, {'max_depth': 36, 'n_estimators': 256}
#  16      0.80197, {'max_depth': 64, 'n_estimators': 64}
#  17      0.80760, {'max_depth': 64, 'n_estimators': 128}
#  18      0.80901, {'max_depth': 64, 'n_estimators': 256}

# model = LogisticRegression(solver='liblinear', C=10, penalty='l1', max_iter=5000)

model = RandomForestClassifier(max_depth=4, n_estimators=256, random_state=1)

model.fit(X_train, y_train.values.ravel())
predictions = model.predict(transformed_X_testing_df)

output = pd.DataFrame({'PassengerId': test_data_df.PassengerId, 'Survived': predictions})
output.to_csv('submission_titanic.csv', index=False)
print("Your submission was successfully saved!")