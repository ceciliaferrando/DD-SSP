import torch
import torch.nn as nn

# base linear regression model, valid for both sgd and dpsgd
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

# base logistic regression model, valid for both sgd and dpsgd
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))