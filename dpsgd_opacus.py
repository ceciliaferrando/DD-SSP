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
