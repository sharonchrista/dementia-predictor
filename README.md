# Dementia Prediction App

This web application predicts whether a patient is *Demented* or *Non-demented* based on clinical parameters using a trained PyTorch neural network model.

## Demo

- Hosted on Hugging Face Spaces: [Live App](https://huggingface.co/spaces/sharonchrisa/dementia-predictor) *(update with real URL)*
- Upcoming Railway Deployment: (add once deployed)

---

## Input Features

The model uses the following 8 clinical inputs:

| Feature | Description |
|--------|-------------|
| M/F    | 0 = Female, 1 = Male |
| Age    | Age of patient |
| EDUC   | Years of education |
| SES    | Socioeconomic status |
| MMSE   | Mini-Mental State Examination score |
| eTIV   | Estimated Total Intracranial Volume |
| nWBV   | Normalized Whole Brain Volume |
| ASF    | Atlas Scaling Factor |

M/F = 1
Age = 78
EDUC = 12
SES = 2
MMSE = 26
eTIV = 1548
nWBV = 0.75
ASF = 1.21

---

## Model Architecture

- Framework: PyTorch
- Model: `DementiaClassifier(input_dim=8)`
- Trained with Binary Cross-Entropy
- Outputs: Probability of Dementia

---

## Technologies Used

- Python 3.10
- Flask / Gradio
- PyTorch
- joblib
- NumPy
- CSV logging
- Hugging Face Spaces / Railway for deployment

---

## Files in This Repo
dementia-predictor/
├── app.py # Main web app
├── model.py # Neural net architecture
├── predict_utils.py # Preprocessing and prediction utils
├── model.pt # Trained model weights
├── scaler.pkl # Scaler for input normalization
├── requirements.txt # All required libraries
├── Procfile # For Railway deployment
├── README.md # Project documentation


---

## How to Run Locally

```bash
git clone https://github.com/yourusername/dementia-predictor.git
cd dementia-predictor
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000 in your browser.

## Future Work
Add REST API interface
Collect feedback from predictions
Add interpretability with SHAP
Add logging to SQLite or Firebase

Acknowledgments
Dataset: Kaggle Dementia Prediction Dataset

License
MIT License.
Model trained on anonymized public dataset.

