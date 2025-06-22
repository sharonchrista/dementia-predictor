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

