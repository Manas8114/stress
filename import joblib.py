import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Load the dataset
slf_df = pd.read_csv('./data/StressLevelDataset.csv')

# Data Exploration
print(slf_df.head())
print(slf_df.info())
print(slf_df.describe().round(2))
print(slf_df.isna().sum())

# Data Visualization
plt.rcParams['figure.figsize'] = [15, 6]
r = 3
c = 7
it = 1
for i in slf_df.columns:
    plt.subplot(r, c, it)
    if slf_df[i].nunique() > 6:
        sns.kdeplot(slf_df[i])
        plt.grid()
    else:
        sns.countplot(x=slf_df[i])
    it += 1
plt.tight_layout()
plt.show()

# Feature Analysis
plt.figure(figsize=(12, 8))
sns.boxplot(x='academic_performance', y='anxiety_level', data=slf_df, showfliers=False)
plt.title('Anxiety levels based on Academic Performance')
plt.show()

# Model Training and Evaluation
X = np.array(slf_df.drop(columns=['stress_level']))
y = np.array(slf_df['stress_level'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# K-Nearest Neighbors (KNN) Classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_predict_train = clf.predict(X_train)
print(f'KNN Train Accuracy: {accuracy_score(y_train, y_predict_train).round(2) * 100}%')

y_predict_test = clf.predict(X_test)
print(f'KNN Test Accuracy: {accuracy_score(y_test, y_predict_test).round(2) * 100}%')

# Confusion Matrix for KNN
training_cm = confusion_matrix(y_train, y_predict_train, labels=clf.classes_)
ConfusionMatrixDisplay(training_cm, display_labels=clf.classes_).plot()
plt.title("Confusion Matrix for Training Data (KNN & 3 Neighbors)")
plt.show()

testing_cm = confusion_matrix(y_test, y_predict_test, labels=clf.classes_)
ConfusionMatrixDisplay(testing_cm, display_labels=clf.classes_).plot()
plt.title("Confusion Matrix for Testing Data (KNN & 3 Neighbors)")
plt.show()

# Decision Tree Classifier
dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
print(f'Decision Tree Train Accuracy: {accuracy_score(y_train, y_train_pred).round(2) * 100}%')

y_pred = dt.predict(X_test)
print(f'Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred).round(2) * 100}%')

# Feature Importance with Decision Tree
fi = dt.feature_importances_
fi_df = pd.DataFrame({'feature': list(slf_df.columns), 'importance': fi}).sort_values('importance', ascending=False)
print(fi_df)

plt.figure(figsize=(15, 10))
tree.plot_tree(dt, filled=True, feature_names=list(slf_df.columns), class_names=['Low', 'Medium', 'High'])
plt.show()
# Model Training and Evaluation
X = np.array(slf_df.drop(columns=['stress_level']))
y = np.array(slf_df['stress_level'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# K-Nearest Neighbors (KNN) Classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_predict_train = clf.predict(X_train)
print(f'KNN Train Accuracy: {accuracy_score(y_train, y_predict_train).round(2) * 100}%')

y_predict_test = clf.predict(X_test)
print(f'KNN Test Accuracy: {accuracy_score(y_test, y_predict_test).round(2) * 100}%')

# Confusion Matrix for KNN
training_cm = confusion_matrix(y_train, y_predict_train, labels=clf.classes_)
ConfusionMatrixDisplay(training_cm, display_labels=clf.classes_).plot()
plt.title("Confusion Matrix for Training Data (KNN & 3 Neighbors)")
plt.show()

testing_cm = confusion_matrix(y_test, y_predict_test, labels=clf.classes_)
ConfusionMatrixDisplay(testing_cm, display_labels=clf.classes_).plot()
plt.title("Confusion Matrix for Testing Data (KNN & 3 Neighbors)")
plt.show()

# Decision Tree Classifier
dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
print(f'Decision Tree Train Accuracy: {accuracy_score(y_train, y_train_pred).round(2) * 100}%')

y_pred = dt.predict(X_test)
print(f'Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred).round(2) * 100}%')

# Feature Importance with Decision Tree
fi = dt.feature_importances_
fi_df = pd.DataFrame({'feature': list(slf_df.columns), 'importance': fi}).sort_values('importance', ascending=False)
print(fi_df)

plt.figure(figsize=(15, 10))
tree.plot_tree(dt, filled=True, feature_names=list(slf_df.columns), class_names=['Low', 'Medium', 'High'])
plt.show()