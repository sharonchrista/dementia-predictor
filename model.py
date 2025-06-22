# dementia_app/modeling/model.py

import torch
import torch.nn as nn

class DementiaClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DementiaClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)
