import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from model import ConcreteGNN  # Import the model definition

# ============================
# 1. Load, Normalize, and Split Data
# ============================
df = pd.read_csv('concrete.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).flatten()

# Save the scalers for later inference
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

df_norm = pd.DataFrame(np.hstack((X_scaled, y_scaled.reshape(-1, 1))), columns=df.columns)
train_df, temp_df = train_test_split(df_norm, test_size=0.3, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

# ============================
# 2. Create Graph Data
# ============================
def create_graph_data(df, k=5):
    features = df.iloc[:, :-1].values
    targets  = df.iloc[:, -1].values
    dist_matrix = squareform(pdist(features, metric='euclidean'))
    edge_index, edge_attr = [], []
    for i in range(len(features)):
        if len(features) == 1:
            break
        neighbors = np.argsort(dist_matrix[i])[1:k+1]
        for j in neighbors:
            edge_index.append([i, j])
            edge_attr.append([dist_matrix[i, j]])
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(targets, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

train_data = create_graph_data(train_df)
val_data   = create_graph_data(val_df)
test_data  = create_graph_data(test_df)

# ============================
# 3. Training & Evaluation
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_data = train_data.to(device)
val_data   = val_data.to(device)
test_data  = test_data.to(device)

train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
val_loader   = DataLoader([val_data], batch_size=1)
test_loader  = DataLoader([test_data], batch_size=1)

model = ConcreteGNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            val_loss += loss.item()
            val_preds.extend(pred.cpu().numpy())
            val_trues.extend(batch.y.squeeze().cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    r2 = r2_score(val_trues, val_preds)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | R2: {r2:.4f}")

torch.save(model.state_dict(), 'concrete_gnn_model.pth')
print("Model saved to concrete_gnn_model.pth")

model.eval()
test_preds = []
test_trues = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_preds.extend(pred.cpu().numpy())
        test_trues.extend(batch.y.squeeze().cpu().numpy())

test_mse = criterion(torch.tensor(test_preds), torch.tensor(test_trues)).item()
test_mae = mean_absolute_error(test_trues, test_preds)
test_r2  = r2_score(test_trues, test_preds)
print(f"\nTest MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")

test_preds_orig = scaler_y.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
test_trues_orig = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()

print("\nSample Predictions (Original Scale):")
for pred_val, true_val in zip(test_preds_orig[:5], test_trues_orig[:5]):
    print(f"Predicted: {pred_val:.2f} MPa, Actual: {true_val:.2f} MPa")
