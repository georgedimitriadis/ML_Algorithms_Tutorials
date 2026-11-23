# Tutorial https://www.youtube.com/watch?v=q90UDEgYqeI
# Data from https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
# saved into ./data/cleveland_heart_disease.csv'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('data/cleveland_heart_disease.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd']

df_no_missing = df.loc[(df['ca']!= '?') & (df['thal'] != '?')]
X = df_no_missing.drop('hd', axis=1).copy()
y = df_no_missing['hd'].copy()
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
y_not_zero_index = y>0
y[y_not_zero_index] = 1

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

classifier = DecisionTreeClassifier(criterion='gini', random_state=42)
classifier.fit(X_train, y_train)
plt.figure(figsize=(50,20))
plot_tree(classifier, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)

ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)

path = classifier.cost_complexity_pruning_path(X_train, y_train)
cpp_alphas = path.ccp_alphas[:-1]

classifier_dts = []

for cpp_alpha in cpp_alphas:
    classifier_dt = DecisionTreeClassifier(criterion='gini', random_state=0, ccp_alpha=cpp_alpha)
    classifier_dt.fit(X_train, y_train)
    classifier_dts.append(classifier_dt)

train_scores = [cls_dt.score(X_train, y_train) for cls_dt in classifier_dts]
test_scores = [cls_dt.score(X_test, y_test) for cls_dt in classifier_dts]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.plot(cpp_alphas, train_scores, label='train', marker='o')
ax.plot(cpp_alphas, test_scores, label='test', marker='o')
ax.legend()

cls_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
scores = cross_val_score(cls_dt, X_train, y_train, cv=5, scoring='accuracy')
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy')


alpha_loop_values = []
for cpp_alpha in cpp_alphas:
    classifier_dt = DecisionTreeClassifier(criterion='gini', random_state=0, ccp_alpha=cpp_alpha)
    scores = cross_val_score(classifier_dt, X_train, y_train, cv=5, scoring='accuracy')
    alpha_loop_values.append([cpp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(data=alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std_accuracy'])

alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std_accuracy', marker='o')

ideal_ccp_alpha = float(alpha_results[(alpha_results['alpha'] > 0.014) & (alpha_results['alpha']< 0.015)]['alpha'].iloc[0])

clf_dt_prunned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
clf_dt_prunned = clf_dt_prunned.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(clf_dt_prunned, X_test, y_test)


plt.figure(figsize=(15,7.5))
plot_tree(clf_dt_prunned, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)