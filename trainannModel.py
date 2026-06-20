# train_ann.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from trainann import ConcreteANN  # Import the model definition only

# ============================
# 1. Load, Normalize, and Split Data
# ============================
df = pd.read_csv('concrete.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).flatten()

# Save scalers
# joblib.dump(scaler_X, "scaler_X.pkl")
# joblib.dump(scaler_y, "scaler_y.pkl")
# print("Scalers saved successfully.")

# Create a DataFrame for normalized data
df_norm = pd.DataFrame(np.hstack((X_scaled, y_scaled.reshape(-1, 1))), columns=df.columns)

# Split into training, validation, and test sets
train_df, temp_df = train_test_split(df_norm, test_size=0.3, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert to numpy arrays (float32)
X_train = train_df.iloc[:, :-1].values.astype(np.float32)
y_train = train_df.iloc[:, -1].values.astype(np.float32)
X_val   = val_df.iloc[:, :-1].values.astype(np.float32)
y_val   = val_df.iloc[:, -1].values.astype(np.float32)
X_test  = test_df.iloc[:, :-1].values.astype(np.float32)
y_test  = test_df.iloc[:, -1].values.astype(np.float32)

# Convert to torch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).view(-1, 1)
X_val   = torch.tensor(X_val)
y_val   = torch.tensor(y_val).view(-1, 1)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test).view(-1, 1)

# ============================
# 2. Define and Train the ANN Model
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_ann = ConcreteANN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_ann.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

num_epochs = 200
for epoch in range(num_epochs):
    model_ann.train()
    optimizer.zero_grad()
    X_train_dev = X_train.to(device)
    y_train_dev = y_train.to(device)
    pred_train = model_ann(X_train_dev)
    loss_train = criterion(pred_train, y_train_dev)
    loss_train.backward()
    optimizer.step()
    
    model_ann.eval()
    with torch.no_grad():
        X_val_dev = X_val.to(device)
        y_val_dev = y_val.to(device)
        pred_val = model_ann(X_val_dev)
        loss_val = criterion(pred_val, y_val_dev)
    scheduler.step(loss_val)
    r2_val = r2_score(y_val_dev.cpu().numpy(), pred_val.cpu().numpy())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f} | R2: {r2_val:.4f}")

# ============================
# 3. Save the Trained Model
# ============================
# torch.save(model_ann.state_dict(), 'ann_model.pth')
# print("ANN model saved to ann_model.pth")
