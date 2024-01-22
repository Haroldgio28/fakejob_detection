
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import pickle

from imblearn.over_sampling import SMOTE


# ## Load Data
df = pd.read_csv('./data/fake_job_postings.csv')


# ## Cleaning
# ### Remove unnecessary columns
del df['job_id']


# ### Location standarizing
# Fill null values on location with "No location"
df['location'].fillna("No location", inplace=True)

# Extract from location the country
df['country'] = df['location'].apply(lambda x: x.split(", ")[0])
df['country'].head()

del df['location']


# ## Filling missing values
del df['salary_range']


## When benefits is not null, then fill with 1, other scenario fill with 0 value
def fill_missing_values(x):
    if type(x) == str:
        return 1
    else:
        return 0

df['benefits'] = df['benefits'].apply(fill_missing_values)
df['company_profile'] = df['company_profile'].apply(fill_missing_values)
df['description'] = df['description'].apply(fill_missing_values)
df['requirements'] = df['requirements'].apply(fill_missing_values)


# ### Handling Missing Values
df['department'].fillna("No registered", inplace=True)
df['required_education'].fillna("No registered", inplace=True)
df['required_experience'].fillna("No registered", inplace=True)
df['industry'].fillna("No registered", inplace=True)
df['function'].fillna("No registered", inplace=True)
df['employment_type'].fillna("No registered", inplace=True)


# ## Split dataframe
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


y_train = df_train.fraudulent
y_val = df_val.fraudulent
y_test = df_test.fraudulent

del df_train['fraudulent']
del df_val['fraudulent']
del df_test['fraudulent']


def mutual_info_fraudulent_score(series):
    return mutual_info_score(series, df_full_train.fraudulent)


col = df_train.columns[1:].tolist()


mi = df_full_train[col].apply(mutual_info_fraudulent_score)


cols_final = mi.sort_values(ascending=False).index[0:10]


df_train = df_train[cols_final]
df_val = df_val[cols_final]
df_test = df_test[cols_final]


# ## Transformation
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


# ## Unbalance handling


# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Saving the results
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'ROC AUC Score': roc_auc}

# Printing results
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()


# The model with best ROC AUC score is SVM.So, we proceed to visualize the ROC curve for it.


svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# Area under the curve (AUC)
# roc_auc = auc(fpr, tpr)

# # ROC Curve Generation
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC AUC Curve')
# plt.legend(loc="lower right")
# plt.show()


svm_model = SVC()

# Hyperparameters grid
param_grid = {
    'C': [0.1, 1],
    'gamma': ['scale', 'auto' ]
}

# Grid search
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best hyperparameters: ", best_params)

best_c = best_params['C']
best_gamma = best_params['gamma']

svm_model = SVC(C=best_c, gamma=best_gamma)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluation metrics
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))


# ## Save the model
output_file = 'model.bin'

f_out = open(output_file, 'wb')
pickle.dump((dv, svm_model), f_out)
f_out.close()

