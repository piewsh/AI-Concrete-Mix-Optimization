import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# ============================
# 1. Load and Normalize Data
# ============================
# "concrete.csv" should have 8 feature columns:
# [Cement, Blast-Furnace Slag, Fly Ash, Water, Superplasticizer,
#  Coarse Aggregate, Fine Aggregate, Age]
# and 1 target column: [Concrete compressive strength].

df = pd.read_csv('concrete.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).flatten()

# Save the scalers for later inference
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("Scalers saved successfully.")

# ============================
# 2. Split Data into Train, Validation, Test
# ============================
# We use a 70/15/15 split.
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ============================
# 3. Train XGBoost Model
# ============================
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# ============================
# 4. Evaluate the Model
# ============================
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print("\nXGBoost Model Evaluation (Normalized Target):")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2:  {r2:.4f}")

# Save the trained XGBoost model
joblib.dump(xgb_model, "xgb_model.pkl")
print("XGBoost model saved successfully as 'xgb_model.pkl'.")

# ============================
# 5. Inverse-transform Sample Predictions to Original Scale
# ============================
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

print("\nXGBoost Sample Predictions (Original Scale):")
for pred_val, true_val in zip(y_pred_orig[:5], y_test_orig[:5]):
    print(f"Predicted: {pred_val:.2f} MPa, Actual: {true_val:.2f} MPa")
