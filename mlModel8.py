import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load data
scopus_data = pd.read_csv('full_data.csv') # get the data from clean.ipynb
scopus2024_data = pd.read_csv('scopus_data.csv')

#--------------------------Function------------------------------------------------------
# Create a function to split subject areas and increment the count
def separate_subject_areas(df, subject_column, count_column):
    # Create an empty DataFrame to store the results
    expanded_rows = []

    for _, row in df.iterrows():
        # Split the combined subject areas (assuming they are separated by commas)
        subject_areas = re.split(r", | and ", row[subject_column])
        
        # For each subject area, create a new row with the same count
        for subject in subject_areas:
            expanded_row = row.copy()
            expanded_row[subject_column] = subject  # Assign the individual subject
            expanded_rows.append(expanded_row)
    
    # Convert the expanded rows into a new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Return the new DataFrame with split subject areas
    return expanded_df

#--------------------------Clean data for ML------------------------------------------------------
scopus2024_aggregated = scopus2024_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',        # Count of publications
    'citedby_count': 'sum'   # Total citations
}).reset_index()

# Filter data for trends (2018–2023)
trend_data = scopus_data[(scopus_data['date_sort_year'] >= 2018) & (scopus_data['date_sort_year'] <= 2023)]

# Train-Test Split before Aggregation
train_data, val_data = train_test_split(trend_data, test_size=0.2, random_state=42)

# Aggregate training data
train_aggregated = train_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',        # Number of publications
    'citedby_count': 'sum'   # Total citations
}).reset_index()

val_aggregated = val_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',        # Number of publications
    'citedby_count': 'sum'   # Total citations
}).reset_index()

#--------------------------Define X & Y------------------------------------------------------
# Features and Target
X_train = train_aggregated[['date_sort_year', 'title']].fillna(0)
y_train = train_aggregated['citedby_count']

X_val = val_aggregated[['date_sort_year', 'title']].fillna(0)
y_val = val_aggregated['citedby_count']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

#--------------------------ML train------------------------------------------------------
# Model Training with Hyperparameter Tuning
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],    
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Validation Predictions
y_val_pred = best_model.predict(X_val_scaled)

# Prepare 2024 Data
X_2024 = scopus2024_aggregated[['date_sort_year', 'title']].fillna(0)
X_2024_scaled = scaler.transform(X_2024)

# Apply log transformation to reduce extreme values for training and prediction
y_train_log = np.log1p(y_train)  # log(1 + x) to handle zero values
y_val_log = np.log1p(y_val)

# Train the model with log-transformed target
best_model.fit(X_train_scaled, y_train_log)

# Prepare and transform the test data similarly
predictions_log = best_model.predict(X_2024_scaled)
predictions_2024 = np.expm1(predictions_log)  # Convert from log scale

# Add predictions to the 2024 dataset
scopus2024_aggregated['predicted_citedby_count'] = predictions_2024
scopus2024_aggregated['predicted_citedby_count'] = scopus2024_aggregated['predicted_citedby_count'].astype(int)

# Check top trending subject areas
top_trending = scopus2024_aggregated.groupby('subject_area')['predicted_citedby_count'].sum().reset_index()
top_trending = top_trending.sort_values(by='predicted_citedby_count', ascending=False)

# Show top trending subject areas for 2024
print("Top Trending Areas for 2024:")
print(top_trending.head(10))

# Recalculate percentage error for comparison
comparison = scopus2024_aggregated.groupby('subject_area').agg({
    'citedby_count': 'sum',
    'predicted_citedby_count': 'sum'
}).reset_index()

# Apply percentage error calculation
comparison['percentage_error'] = (
    abs(comparison['citedby_count'] - comparison['predicted_citedby_count'])
    / comparison['citedby_count'].replace(0, np.nan)
) * 100
comparison['percentage_error'] = comparison['percentage_error'].fillna(0)

# Drop rows where both citedby_count and predicted_citedby_count are 0
comparison = comparison[(comparison['citedby_count'] != 0) | (comparison['predicted_citedby_count'] != 0)]
# Apply the function to separate the subject areas and update the DataFrame
comparison = separate_subject_areas(comparison, 'subject_area', 'citedby_count')

# Group by the column(s) where duplicate names appear and aggregate numerical values
combined_data = comparison.groupby('subject_area', as_index=False).agg({
    'citedby_count': 'sum',                # Sum citation counts
    'predicted_citedby_count': 'sum',      # Sum predicted counts
    'percentage_error': 'mean'             # Average percentage error
})

print("Comparison of Actual vs Predicted (Top 10):")
print(combined_data.sort_values(by='citedby_count', ascending=False).head(10))

#--------------------------Visualization------------------------------------------------------
# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['citedby_count'], combined_data['predicted_citedby_count'], alpha=0.6, label="Data Points")
plt.plot([0, max(combined_data['citedby_count'])], [0, max(combined_data['citedby_count'])], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Citedby Count")
plt.ylabel("Predicted Citedby Count")
plt.title("Actual vs Predicted Citation Counts (2024)")
plt.legend()
plt.show()

#--------------------------Score parts------------------------------------------------------
# Metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation Metrics:\nMSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# Ensure 'citedby_count' and 'predicted_citedby_count' are numeric, converting invalid values to NaN
combined_data['citedby_count'] = pd.to_numeric(combined_data['citedby_count'], errors='coerce')
combined_data['predicted_citedby_count'] = pd.to_numeric(combined_data['predicted_citedby_count'], errors='coerce')

# Define bins for citation counts
bins = [0, 100, 500, np.inf]
labels = ['Low', 'Medium', 'High']

# Bin the actual and predicted values
actual_binned = pd.cut(combined_data['citedby_count'], bins=bins, labels=labels)
predicted_binned = pd.cut(combined_data['predicted_citedby_count'], bins=bins, labels=labels)

# Ensure that the binned values are treated as strings for comparison
actual_binned = actual_binned.astype(str)
predicted_binned = predicted_binned.astype(str)

# Calculate precision, recall, and F1 score
precision = precision_score(actual_binned, predicted_binned, average='weighted', labels=labels, zero_division=0)
recall = recall_score(actual_binned, predicted_binned, average='weighted', labels=labels, zero_division=0)
f1 = f1_score(actual_binned, predicted_binned, average='weighted', labels=labels, zero_division=0)

# Print the results
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

#--------------------------Save to csv------------------------------------------------------
# Save results to CSV
combined_data.to_csv('ML_pred.csv', index=False)