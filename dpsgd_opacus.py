import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def prepare_data_loaders(X, y, X_test, y_test, batch_size=32, scale_data=False, shuffle=False):

    if scale_data:
        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy().flatten(), dtype=torch.float32)
    train_dataset = TensorDataset(X_tensor, y_tensor)

    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy().flatten(), dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


# logreg model with a single dense layer and sigmoid activation
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Trainer class for handling the training process
class Opacus_DPSGD:
    def __init__(self, model, train_loader, test_loader, epochs, epsilon, delta, lr, max_grad_norm, use_dp=True):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.epsilon = epsilon
        self.delta = delta
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.use_dp = use_dp
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = self.model.criterion
        self.sample_size = len(self.train_loader.dataset)
        self.batch_size = self.train_loader.batch_size
        self.noise_multiplier = None
        self.losses = []
        self.grad_norms = []

        if self.use_dp:
            self._setup_dp()

    def _setup_dp(self):
        # Calculate the noise multiplier
        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            sample_rate=self.batch_size / self.sample_size,
            epochs=self.epochs
        )
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        print(f"Using DP with noise multiplier: {self.noise_multiplier}")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                if self.use_dp:
                    self._dp_step()
                else:
                    self.optimizer.step()

                # Track the running loss for this batch
                running_loss += loss.item()

            # Track the average loss for the epoch
            epoch_loss = running_loss / len(self.train_loader)
            self.losses.append(epoch_loss)  # Store the loss for plotting
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def _dp_step(self):
        # Clip the gradients for each parameter in place
        for param in self.model.parameters():
            if param.grad is not None:
                clip_grad_norm_([param], self.max_grad_norm)

        # Add noise to the gradients
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape
                ).to(param.device)
                param.grad += noise

        self.optimizer.step()

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                predicted = outputs.round()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total
        print(f"Test Loss: {running_loss / len(self.test_loader):.4f}, Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def auc(self):
        self.model.eval()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)

                # Collect targets and predicted probabilities for AUC calculation
                all_targets.extend(targets.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())

        # Calculate AUC
        auc = roc_auc_score(all_targets, all_outputs)
        print(f"Test AUC: {auc:.4f}")
        return auc

    def plot_metrics(self):
        # Plotting loss and gradient norms
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses, color=color, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Gradient Norm', color=color)
        ax2.plot(self.grad_norms, color=color, label='Gradient Norm')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Training Loss and Gradient Norm')
        fig.tight_layout()
        plt.show()


# def train_dpsgd_manual_opacus(model, train_loader, epochs, epsilon, delta,
#                                lr, max_grad_norm, use_dp=True):
#
#     # initialize optimizer (this will be wrapped by an Opacus optimizer)
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#
#     # get sample and batch size from loader
#     sample_size = len(train_loader.dataset)
#     batch_size = train_loader.batch_size
#
#     # loss criterion
#     criterion = nn.BCELoss()
#
#     # use Opacus to calculate the noise multiplier if not provided
#     if use_dp:
#         noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
#                                                 sample_rate=batch_size / sample_size,
#                                                 epochs=epochs)
#         privacy_engine = PrivacyEngine()
#         model, optimizer, train_loader = privacy_engine.make_private(
#             module=model,
#             optimizer=optimizer,
#             data_loader=train_loader,
#             noise_multiplier=noise_multiplier,
#             max_grad_norm=max_grad_norm
#         )
#
#         print("NOISE MULTIPLIER", noise_multiplier)
#
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         # iterate over the batches in the dataset
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.float(), targets.float()
#             targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)
#
#             # reset the gradients (otherwise they accumulate across batches)
#             optimizer.zero_grad()
#
#             # forward pass from input
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#
#             # Backward pass
#             loss.backward()
#
#             if use_dp:
#                 # Clip the gradients for each parameter in place
#                 for param in model.parameters():
#                     if param.grad is not None:
#                         clip_grad_norm_([param], max_grad_norm)
#
#                 # Add noise to each gradient and apply the update
#                 for param in model.parameters():
#                     if param.grad is not None:
#                         noise = torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=param.grad.shape).to(param.device)
#                         param.grad += noise
#
#             # Perform the optimization step
#             optimizer.step()
#
#             # Accumulate loss
#             running_loss += loss.item()
#
#             # Calculate accuracy
#             predicted = outputs.round()  # Convert probabilities to binary predictions
#             correct += (predicted == targets).sum().item()
#             total += targets.size(0)
#
#         # for batch_idx, (inputs, targets) in enumerate(train_loader):
#         #     inputs, targets = inputs.float(), targets.float()
#         #     targets = targets.unsqueeze(1)  # Reshape targets to match model output (batch_size, 1)
#         #
#         #     optimizer.zero_grad()   # check
#         #     outputs = model(inputs)
#         #
#         #     loss = criterion(outputs, targets)
#         #     loss.backward()
#         #
#         #     optimizer.step()
#         #
#         #     # Accumulate metrics
#         #     running_loss += loss.item()
#         #     predicted = outputs.round()
#         #     correct += (predicted == targets).sum().item()
#         #     total += targets.size(0)
#
#         # Print epoch statistics
#         accuracy = correct / total
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")
#
#     return model