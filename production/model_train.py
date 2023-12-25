# %%
import argparse
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
# parser = argparse.ArgumentParser()
# parser.add_argument("--datapath", type=str, required=True, help='Path to training dataset')
# args = parser.parse_args()
# df_train_data = pd.read_csv(args.datapath)

df_train_data = pd.read_csv('../train.csv')
df_train_data.head()

# %%
df_train_data.tail()

# %%
df_train_data.shape

# %%
df_train_data.isnull().sum()

# %%
df_train_data['Activity'].unique()

# %%
activity_counts = df_train_data['Activity'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Activities')
plt.show()

# %%
df_train_data['subject'].unique()

# %%
# X = pd.DataFrame(df_train_data.drop(['Activity', 'subject'], axis=1))
# y = df_train_data.Activity.values.astype(object)

# X.shape, y.shape

X = df_train_data.drop(['Activity', 'subject'], axis=1)
y = df_train_data['Activity'].values.astype(object)

print("X shape:", X.shape)
print("y shape:", y.shape)

# %%
X.head()

# %%
y[1000]

# %%
X.info()

# %%
num_of_cols = X.select_dtypes(include='number').columns
print("Number of numeric features:", num_of_cols.size) 

# %%
y = preprocessing.LabelEncoder().fit_transform(y)
print("Shape of y:", y.shape)

# %%
y[5], encoder.classes_

# %%
encoder.classes_[5]

# %%
standard_scaler = StandardScaler()

X = standard_scaler.fit_transform(X)
X[5]

# %%


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
def eval_model(true, predicted):
    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted, average='weighted', zero_division=1)
    recall = recall_score(true, predicted, average='weighted', zero_division=1)
    f1 = f1_score(true, predicted, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(true, predicted)
    
    return accuracy, precision, recall, f1, conf_matrix


# %%
models = {
    # "SVM": SVC(),
    "SVM": SVC(kernel='rbf',C=100.0),
    # "Logistic Regression": LogisticRegression(random_state=5, max_iter=2000),
}

list_of_model = []
accuracy_list =[]

# %%
for i in range(len(list(models))):
    model = list(models.values())[i]
    
    # Train model
    model.fit(X_train, y_train) 

    # Make predictions
    y_train_prediction = model.predict(X_train)
    y_test_prediction = model.predict(X_test)

    model_train_accuracy , model_train_precision, model_train_recall, model_train_f1, model_train_cm = eval_model(y_train, y_train_prediction)

    model_test_accuracy , model_test_precision, model_test_recall, model_test_f1, model_test_cm = eval_model(y_test, y_test_prediction)

    print(list(models.keys())[i])
    list_of_model.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print("- Precision: {:.4f}".format(model_train_precision))
    print("- Recall: {:.4f}".format(model_train_recall))
    print("- f1: {:.4f}".format(model_train_f1))
 

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Accuracy: {:.4f}".format(model_test_accuracy))
    print("- Precision: {:.4f}".format(model_test_precision))
    print("- Recall: {:.4f}".format(model_test_recall))
    print("- f1: {:.4f}".format(model_test_f1))
    
    accuracy_list.append(model_test_accuracy)
    
    print('='*35)
    print('\n')


# %%
result_df = pd.DataFrame(list(zip(list_of_model, accuracy_list)), columns=['Algorithm_Type', 'Accuracy_Score']).sort_values(by=["Accuracy_Score"],ascending=False)
print(result_df)


