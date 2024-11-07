import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch.nn.utils import clip_grad_norm_
import numpy as np


# logreg model with a single dense layer and sigmoid activation
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_dpsgd_manual_opacus(model, train_loader, epochs, epsilon, delta,
                               lr, max_grad_norm, use_dp=True):

    # initialize optimizer (this will be wrapped by an Opacus optimizer)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # get sample and batch size from loader
    sample_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size

    # loss criterion
    criterion = nn.BCELoss()

    # use Opacus to calculate the noise multiplier if not provided
    if use_dp:
        noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                                sample_rate=batch_size / sample_size,
                                                epochs=epochs)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )

        print("NOISE MULTIPLIER", noise_multiplier)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # iterate over the batches in the dataset
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.float(), targets.float()
            targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)

            # reset the gradients (otherwise they accumulate across batches)
            optimizer.zero_grad()

            # forward pass from input
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            if use_dp:
                # Clip the gradients for each parameter in place
                for param in model.parameters():
                    if param.grad is not None:
                        clip_grad_norm_([param], max_grad_norm)

                # Add noise to each gradient and apply the update
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=param.grad.shape).to(param.device)
                        param.grad += noise

            # Perform the optimization step
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            predicted = outputs.round()  # Convert probabilities to binary predictions
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        # for batch_idx, (inputs, targets) in enumerate(train_loader):
        #     inputs, targets = inputs.float(), targets.float()
        #     targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)
        #
        #     optimizer.zero_grad()   # check
        #     outputs = model(inputs)
        #
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        #
        #     optimizer.step()
        #
        #     # Accumulate metrics
        #     running_loss += loss.item()
        #     predicted = outputs.round()
        #     correct += (predicted == targets).sum().item()
        #     total += targets.size(0)

        # Print epoch statistics
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")

    return model