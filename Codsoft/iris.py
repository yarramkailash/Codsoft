import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
iris_df = pd.read_csv('IRIS.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(iris_df.describe())

# Pairplot to visualize relationships between features
print("\nPairplot to visualize relationships between features:")
# Adjusting the pairplot to include more details
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'D'], palette='husl', height=3)
plt.suptitle('Enhanced Pairplot of Iris Dataset', y=1.02)
plt.show()

# Splitting the dataset into features (X) and labels (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning using GridSearchCV
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': np.arange(3, 12),  # Adjusting the range for max_depth
              'min_samples_split': [2, 3, 4, 5]}  # Adding more options for min_samples_split

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Visualize the decision tree with more details
plt.figure(figsize=(16, 10))
plot_tree(best_model, feature_names=X.columns, class_names=iris_df['species'].unique(),
          filled=True, rounded=True, fontsize=10, max_depth=4)  # Limiting the depth for better visualization
plt.title("Enhanced Decision Tree Visualization")
plt.show()

# Model evaluation with additional metrics
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Additional metrics: precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix with a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Enhanced Confusion Matrix')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
iris_df = pd.read_csv('IRIS.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(iris_df.describe())

# Pairplot to visualize relationships between features
print("\nPairplot to visualize relationships between features:")
# Adjusting the pairplot to include more details
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'D'], palette='husl', height=3)
plt.suptitle('Enhanced Pairplot of Iris Dataset', y=1.02)
plt.show()

# Splitting the dataset into features (X) and labels (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning using GridSearchCV
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': np.arange(3, 12),  # Adjusting the range for max_depth
              'min_samples_split': [2, 3, 4, 5]}  # Adding more options for min_samples_split

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Visualize the decision tree with more details
plt.figure(figsize=(16, 10))
plot_tree(best_model, feature_names=X.columns, class_names=iris_df['species'].unique(),
          filled=True, rounded=True, fontsize=10, max_depth=4)  # Limiting the depth for better visualization
plt.title("Enhanced Decision Tree Visualization")
plt.show()

# Model evaluation with additional metrics
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Additional metrics: precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix with a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Enhanced Confusion Matrix')
plt.show()
