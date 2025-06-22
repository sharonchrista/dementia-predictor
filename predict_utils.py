# dementia_app/predict_utils.py

import torch
import numpy as np
from model import DementiaClassifier


def load_model(model_path, input_dim=8):
    """
    Load the trained PyTorch model from file.

    Parameters:
        model_path (str): Path to the .pt file
        input_dim (int): Number of input features

    Returns:
        model (nn.Module): Loaded and ready model
    """
    model = DementiaClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_dementia(model, input_data):
    """
    Make a prediction using the trained model.

    Parameters:
        model (nn.Module): Trained model
        input_data (list or ndarray): One sample of input data

    Returns:
        int: 0 (Non-demented) or 1 (Demented)
    """
    input_tensor = torch.tensor(np.array([input_data]), dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = int(output.item() > 0.5)
    return prediction


import joblib

def preprocess_input(input_data):
    """
    Preprocess the input data using the saved StandardScaler.
    
    Parameters:
        input_data (list): List of 8 input features
    
    Returns:
        torch.Tensor: Scaled input tensor
    """
    scaler = joblib.load("models/scaler.pkl")  # Load the trained scaler
    input_array = np.array([input_data], dtype=np.float32)  # Shape: (1, 8)
    scaled_input = scaler.transform(input_array)
    return torch.tensor(scaled_input, dtype=torch.float32)

