import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge  # Change the linear regression model to Ridge regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load the movie data with appropriate encoding
df = pd.read_csv('IMDb movies India.csv', encoding='latin1')

# Preprocess data
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown') 
df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown') 
df['Actor 3'] = df['Actor 3'].fillna('Unknown')

# Encode categorical features
label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actor 1'] = label_encoder.fit_transform(df['Actor 1']) 
df['Actor 2'] = label_encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = label_encoder.fit_transform(df['Actor 3'])

# Drop rows with missing target values
df.dropna(subset=['Rating'], inplace=True)

# Create a TfidfVectorizer to vectorize text data
tfidf = TfidfVectorizer(stop_words='english')

# Concatenate text columns into a single document
df['text'] = df['Genre'].astype(str) + ' ' + df['Director'].astype(str) + ' ' + \
             df['Actor 1'].astype(str) + ' ' + df['Actor 2'].astype(str) + ' ' + \
             df['Actor 3'].astype(str)

# Split data
X = df['text']
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline with Ridge regression
model = Pipeline([('tfidf', tfidf),
                  ('ridge_regression', Ridge(alpha=1.0))])  # Set regularization strength (alpha) for Ridge

# Train model
model.fit(X_train, y_train) 

# Evaluate model using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print('Cross-Validation Mean Squared Error:', -cv_scores.mean())

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate evaluation metrics
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print('Train Mean Absolute Error:', train_mae)
print('Test Mean Absolute Error:', test_mae)
print('Train Root Mean Squared Error:', train_rmse)
print('Test Root Mean Squared Error:', test_rmse)

# Provide information about predicted rating for the sample data
sample_data = ['1 5 7 9 2']  # Encoded genre, director, and 3 actors as a string
sample_vectorized = model.named_steps['tfidf'].transform(sample_data) 
predicted_rating = model.named_steps['ridge_regression'].predict(sample_vectorized)
print('Sample Predicted Ratings:', predicted_rating)

# Visualize actual vs. predicted ratings and histogram of predicted ratings
plt.figure(figsize=(18, 8))

# Subplot 1: Scatter plot for Actual vs. Predicted Ratings
plt.subplot(1, 3, 1)
plt.scatter(y_test, test_predictions, color='orange', alpha=0.6)  # Change scatter color to orange
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', lw=2)  # Change line color to green
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs. Predicted Rating')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Bar chart for Actual vs. Predicted Rating for Sample Data
plt.subplot(1, 3, 2)
plt.bar(['Actual', 'Predicted'], [y_test.iloc[0], predicted_rating[0]], color=['lightblue', 'lightcoral'], alpha=0.8)  # Change bar colors
plt.xlabel('Rating Type')
plt.ylabel('Rating')
plt.title('Actual vs. Predicted Rating (Sample Data)')

# Subplot 3: Histogram for Distribution of Actual and Predicted Ratings
plt.subplot(1, 3, 3)
plt.hist(y_test, bins=20, color='skyblue', alpha=0.6, label='Actual Ratings', edgecolor='black')
plt.hist(test_predictions, bins=20, color='lightcoral', alpha=0.6, label='Predicted Ratings', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
