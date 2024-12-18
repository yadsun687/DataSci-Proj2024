import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load data
scopus_data = pd.read_csv('full_data.csv')  # Load your dataset
scopus2024_data = pd.read_csv('scopus_data.csv')

scopus2024_aggregated = scopus2024_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',
    'citedby_count': 'sum'
}).reset_index()

# Filter data for trends (2018–2023)
trend_data = scopus_data[(scopus_data['date_sort_year'] >= 2018) & (scopus_data['date_sort_year'] <= 2023)]

# Train-Test Split before Aggregation
train_data, val_data = train_test_split(trend_data, test_size=0.2, random_state=42)

# Aggregate training data
train_aggregated = train_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',
    'citedby_count': 'sum'
}).reset_index()

val_aggregated = val_data.groupby(['date_sort_year', 'subject_area']).agg({
    'title': 'count',
    'citedby_count': 'sum'
}).reset_index()

# Features and Target
X_train = train_aggregated[['date_sort_year', 'title']].fillna(0)
y_train = train_aggregated['citedby_count']

X_val = val_aggregated[['date_sort_year', 'title']].fillna(0)
y_val = val_aggregated['citedby_count']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

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

# Metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.6, label="Data Points")
plt.plot([0, max(y_val)], [0, max(y_val)], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Citedby Count")
plt.ylabel("Predicted Citedby Count")
plt.title("Actual vs Predicted Citation Counts")
plt.legend()

# Save plot to a BytesIO object
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
plt.close()

# Start MLflow Logging
with mlflow.start_run(run_name="Random Forest Regression") as run:
    # Log Parameters
    mlflow.log_param("n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("random_state", 42)
    
    # Log Metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log Model
    mlflow.sklearn.log_model(best_model, "model", registered_model_name="RandomForestRegressor")

    # Save the plot to a file
    plot_path = "rf_plot.png"
    with open(plot_path, "wb") as f:
        f.write(img.getvalue())

    # Log Plot as Artifact
    mlflow.log_artifact(plot_path)

    # Output run ID for reference
    print(f"Run ID: {run.info.run_id}")

# Clean up the plot file
import os
os.remove(plot_path)
