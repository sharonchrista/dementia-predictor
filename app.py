import gradio as gr
import torch
import numpy as np
import os
import joblib
import datetime
import csv
import sys

from model import DementiaClassifier
from predict_utils import preprocess_input, predict_dementia, load_model

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = DementiaClassifier(input_dim=8)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Prediction function
def predict(MF, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF):
    try:
        # Convert and scale input
        inputs = np.array([[MF, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF]])
        scaled_inputs = scaler.transform(inputs)
        tensor_input = torch.tensor(scaled_inputs, dtype=torch.float32)

        # Get prediction and confidence
        with torch.no_grad():
            output = model(tensor_input)
            prob = torch.sigmoid(output).item()
            prediction = "Demented" if prob > 0.5 else "Non-demented"
            confidence = f"{prob * 100:.2f}%" if prob > 0.5 else f"{(1 - prob) * 100:.2f}%"

        # Optional logging
        with open("gradio_predictions_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                MF, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF,
                prediction, confidence
            ])

        return prediction, confidence

    except Exception as e:
        return f"Error: {str(e)}", ""

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="M/F (0 = Female, 1 = Male)"),
        gr.Number(label="Age"),
        gr.Number(label="EDUC"),
        gr.Number(label="SES"),
        gr.Number(label="MMSE"),
        gr.Number(label="eTIV"),
        gr.Number(label="nWBV"),
        gr.Number(label="ASF"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
    ],
    title="Dementia Prediction App",
    description="Enter the clinical parameters to predict if a patient is demented.",
    theme="default"
)

# Launch
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
