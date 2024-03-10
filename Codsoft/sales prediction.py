import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('advertising.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Visualize the relationships between features and the target variable (Sales)
print("\nVisualizing the relationships between features and the target variable:")

# Create a pairplot with regression line equations and correlation coefficients
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='reg', plot_kws={'line_kws':{'color':'red'}}, diag_kind=None)

# Add correlation coefficients to the plot
corr_matrix = data.corr()
for i, feature1 in enumerate(['TV', 'Radio', 'Newspaper']):
    for j, feature2 in enumerate(['TV', 'Radio', 'Newspaper']):
        if i == j:
            continue
        plt.text(i, j, f"Corr: {corr_matrix.loc[feature1, feature2]:.2f}", ha='center', va='center', fontsize=10, color='blue')

plt.show()

# Pairplot with hue for better visualization
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='reg', hue='Sales', palette='viridis')
plt.suptitle('Enhanced Pairplot with Hue for Sales', y=1.02)
plt.show()

# Correlation heatmap for better visualization
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Splitting the data into train and test sets
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print('\nModel Evaluation:')
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

# Calculate the perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
perfect_line = np.linspace(min_val, max_val, 100)

# Plotting predicted vs actual values with color-coding and perfect prediction line
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, c='blue', alpha=0.7, label='Actual vs Predicted', edgecolors='k')
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'Actual vs Predicted Sales (R^2 = {r2_score(y_test, y_pred):.2f})')
plt.legend()
plt.grid(True)
plt.show()
