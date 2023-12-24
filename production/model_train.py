# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import  preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

# %%
df_train = pd.read_csv('../train.csv')
df_train.head()

# %%
df_train.tail()

# %%
df_train.shape

# %%
df_train.isnull().sum()

# %%
df_train['Activity'].unique()

# %%
activity_counts = df_train['Activity'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Activities')
plt.show()
# plt.figure(figsize=(6,3))

# axis = sns.countplot(x="Activity", data=df_train)
# plt.xticks(x=df_train['Activity'], rotation='vertical')
# plt.show()

# %%
df_train['subject'].unique()

# %%
X = pd.DataFrame(df_train.drop(['Activity', 'subject'], axis=1))
y = df_train.Activity.values.astype(object)

# %%
X.shape, y.shape

# %%
X.head()

# %%
y[100]

# %%
X.info()

# %%
num_cols = X._get_numeric_data().columns
print("Number of numeric features:", num_cols.size) 

# %%
encoder = preprocessing.LabelEncoder()

# %%
encoder.fit(y)
y = encoder.transform(y)
y.shape

# %%
y[5]

# %%
encoder.classes_

# %%
encoder.classes_[5]

# %%
scaler = StandardScaler()

# %%
X = scaler.fit_transform(X)
X[5]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
def evaluate_model(true, predicted):
    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted, average='weighted', zero_division=1)
    recall = recall_score(true, predicted, average='weighted', zero_division=1)
    f1 = f1_score(true, predicted, average='weighted', zero_division=1)
    cm = confusion_matrix(true, predicted)
    
    return accuracy, precision, recall, f1, cm


# %%
models = {
    "Logistic Regression": LogisticRegression(random_state=5, max_iter=2000),
    "SVM": SVC(kernel='rbf',C=100.0),
    # "SVM": SVC(),
}

model_list = []
r2_list =[]

# %%
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_accuracy , model_train_precision, model_train_recall, model_train_f1, model_train_cm = evaluate_model(y_train, y_train_pred)

    model_test_accuracy , model_test_precision, model_test_recall, model_test_f1, model_test_cm = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print("- Precision: {:.4f}".format(model_train_precision))
    print("- Recall: {:.4f}".format(model_train_recall))
    print("- f1: {:.4f}".format(model_train_f1))
    # print("conf matrix {}:".format(model_train_cm))


    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Accuracy: {:.4f}".format(model_test_accuracy))
    print("- Precision: {:.4f}".format(model_test_precision))
    print("- Recall: {:.4f}".format(model_test_recall))
    print("- f1: {:.4f}".format(model_test_f1))
    # print("conf matrix: {}".format(model_test_cm))
    r2_list.append(model_test_accuracy)
    
    print('='*35)
    print('\n')


# %%
pd.DataFrame(list(zip(model_list, r2_list)), columns=['Algorithm_Type', 'Accuracy_Score']).sort_values(by=["Accuracy_Score"],ascending=False)

# %%
# tuned_svc = SVC(kernel='rbf',C=100.0)


