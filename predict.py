import joblib
import torch     
from trainann import ConcreteANN

# ----------------------------
# Page 1: Concrete Performance Prediction
# ----------------------------
def load_scalers():
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return scaler_X, scaler_y

def load_models(device):
    # Load ANN model
    model_ann = ConcreteANN()
    model_ann.load_state_dict(torch.load("ann_model.pth", map_location=device))
    model_ann.to(device)
    model_ann.eval()
    
    # Load XGBoost model (using joblib)
    xgb_model = joblib.load('xgb_model.pkl')
    
    return model_ann, xgb_model

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()
scaler_X, scaler_y = load_scalers()
model_ann, xgb_model = load_models(device)