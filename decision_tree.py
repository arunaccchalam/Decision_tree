import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import plot_tree


# data
df = pd.read_csv("penguins_size.csv")
df.head()

# Missing Data

df.info()

# getting the unfilled rows
df.isna().sum()

# What percentage are we dropping?
100*(10/344)

df = df.dropna()
df.info()
df.head()

df['sex'].unique()

df['island'].unique()

df = df[df['sex']!='.']

# Visualization

sns.scatterplot(x='culmen_length_mm',y='culmen_depth_mm',data=df,hue='species',palette='Dark2')
sns.pairplot(df,hue='species',palette='Dark2')
sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',col='sex',palette='Dark2')

# Feature Engineering

pd.get_dummies(df)
pd.get_dummies(df.drop('species',axis=1),drop_first=True)

# Train | Test Split

X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree Classifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

base_pred = model.predict(X_test)

# Evaluation

confusion_matrix(y_test,base_pred)

plot_confusion_matrix(model,X_test,y_test)

print(classification_report(y_test,base_pred))

model.feature_importances_

pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance'])

sns.boxplot(x='species',y='body_mass_g',data=df)

# Visualize the Tree

plt.figure(figsize=(12,8))
plot_tree(model);

plt.figure(figsize=(12,8),dpi=150)
plot_tree(model,filled=True,feature_names=X.columns);

# Reporting Model Results

def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=150)
    plot_tree(model,filled=True,feature_names=X.columns);

# Understanding Hyperparameters
# Max Depth

help(DecisionTreeClassifier)

pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train,y_train)

report_model(pruned_tree)

# Max Leaf Nodes

pruned_tree = DecisionTreeClassifier(max_leaf_nodes=3)
pruned_tree.fit(X_train,y_train)

report_model(pruned_tree)

# Criterion

entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train,y_train)

report_model(entropy_tree)
