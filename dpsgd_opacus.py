import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch.nn.utils import clip_grad_norm_


# logreg model with a single dense layer and sigmoid activation
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# train the model using opacus dpsgd
def train_dpsgd_opacus(model, train_loader, epochs, epsilon, delta, lr, max_grad_norm):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # # Privacy engine requires batch size, sample size, and steps to compute noise
    sample_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    steps_per_epoch = len(train_loader)

    # Use Opacus to compute the noise multiplier based on epsilon, delta, and the number of training steps
    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=batch_size / sample_size,
                                                epochs=epochs)

    criterion = nn.BCELoss()

    # initialize the privacy engine
    privacy_engine = PrivacyEngine()

    # instantiate model, optimizare, dataloader. "with_epsilon" performs the noise multiplier calculation internally
    # given epsilon and delta
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                                                                            module = model,
                                                                            optimizer = optimizer,
                                                                            data_loader = train_loader,
                                                                            target_epsilon = epsilon,
                                                                            target_delta=delta,
                                                                            epochs=epochs,
                                                                            max_grad_norm = max_grad_norm)


    # Dictionary to track pre-clipping gradients for each parameter
    pre_clipping_gradients = {}

    # Hook to save the pre-clipping gradients
    def save_pre_clipping_gradients(module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                pre_clipping_gradients[name] = param.grad.clone().detach()

    for epoch in range(epochs):
        model.train()   # check this line
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.float(), targets.float()
            targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)

            optimizer.zero_grad()   # check
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            # save pre-clipping gradients using the backward hook
            model.apply(save_pre_clipping_gradients)

            # DP-SGD step
            optimizer.step()

            # loss and accuracy
            running_loss += loss.item()

            predicted = outputs.round()  # Convert probabilities to binary predictions
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")

        # L2 norm of the pre-clipping gradients for each parameter
        for name, grad in pre_clipping_gradients.items():
            l2_norm = grad.norm(2).item()  # Calculate L2 norm (Euclidean norm)
            print(f"L2 norm of pre-clipping gradient for {name}: {l2_norm}")

        # privacy budget used so far
        epsilon_spent = privacy_engine.get_epsilon(delta)
        print(f"Privacy budget spent (ε): {epsilon_spent:.2f}")

    return model

def train_dpsgd_manual_opacus(model, train_loader, epochs, epsilon, delta, lr=0.01, max_grad_norm=1.0, noise_multiplier=None, microbatch_size=1):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Initialize the privacy engine
    sample_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    steps_per_epoch = len(train_loader)
    criterion = torch.nn.BCELoss()

    # Use Opacus to calculate the noise multiplier if not provided
    if noise_multiplier is None:
        noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                                sample_rate=batch_size / sample_size,
                                                         epochs=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.float(), targets.float()
            targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)

            # Reset accumulated gradients for each parameter
            for param in model.parameters():
                param.accumulated_grads = []

            # Split the batch into microbatches
            for i in range(0, len(inputs), microbatch_size):
                microbatch_inputs = inputs[i:i + microbatch_size]
                microbatch_targets = targets[i:i + microbatch_size]

                optimizer.zero_grad()
                outputs = model(microbatch_inputs)
                loss = criterion(outputs, microbatch_targets)
                loss.backward()

                # Clip each parameter's per-sample gradient and accumulate it
                for param in model.parameters():
                    if param.grad is not None:
                        per_sample_grad = param.grad.detach().clone()
                        clip_grad_norm_([per_sample_grad], max_grad_norm)  # In-place clipping
                        param.accumulated_grads.append(per_sample_grad)

            # Aggregate the gradients across the microbatches
            for param in model.parameters():
                if param.accumulated_grads:
                    param.grad = torch.stack(param.accumulated_grads, dim=0).mean(dim=0)

            # Add noise to the gradients and update the parameters
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=param.grad.shape)
                    param.grad += noise
                    param.data -= lr * param.grad

            # Compute loss and accuracy for the batch
            running_loss += loss.item()

            predicted = outputs.round()  # Convert probabilities to binary predictions
            correct += (predicted == microbatch_targets).sum().item()
            total += microbatch_targets.size(0)

        # Print epoch statistics
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")

        # (Optional) Calculate the privacy budget spent so far using Opacus
        # epsilon_spent = privacy_engine.get_epsilon(delta)
        # print(f"Privacy budget spent (ε): {epsilon_spent:.2f}")

    return model
